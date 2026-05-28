# %%
# 2 Instantiating the Model
from src.models.model import SegformerB5

# This downloads ~330 MB of weights from HuggingFace on first run
model = SegformerB5(
    # 14 matches the layout of the pre-baked GeoTIFFs: the 12 L2A spectral bands
    # (B10 is dropped by atmospheric correction) plus NDVI and NDWI as derived
    # channels. `src/download_region.py` produces the same layout. If you swap in
    # a different dataset (different band count or different derived layers), this
    # number, the normalisation statistics, and `num_channels` in the SegformerConfig
    # must all change together — otherwise the first patch-embedding layer
    # mis-shapes inputs.
    n_bands=14,
    logits=True,             # return raw logits (not probabilities)
    freeze_encoder=False,    # keep encoder trainable
    type_labeler="CLCplus-Backbone",
)


# %%
# 3.1 Printing the architecture
print(model)


# %%
# 3.2 Counting parameters
# A quick way to understand a model’s capacity is to count how many parameters it has, and how many are currently trainable (i.e. not frozen).

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(model)
enc_total, enc_trainable = count_params(model.segformer)
dec_total, dec_trainable = count_params(model.decode_head)

print(f"{'Component':<20} {'Total params':>15} {'Trainable params':>18}")
print("-" * 55)
print(f"{'Encoder (MiT-B5)':<20} {enc_total:>15,} {enc_trainable:>18,}")
print(f"{'Decoder (MLP)':<20} {dec_total:>15,} {dec_trainable:>18,}")
print(f"{'Full model':<20} {total:>15,} {trainable:>18,}")


# %%
# 3.3 Freezing the encoder
# When fine-tuning on a new domain (e.g. switching from RGB natural images to 14-channel satellite imagery), a common strategy is to freeze the encoder at first and only train the decoder. This is much faster and prevents overfitting when labelled data is scarce.

# Freeze all encoder parameters
model.freeze()

total, trainable = count_params(model)
print(f"After freezing encoder — trainable: {trainable:,} / {total:,}")

# Unfreeze for full fine-tuning
model.unfreeze()
total, trainable = count_params(model)
print(f"After unfreezing — trainable: {trainable:,} / {total:,}")


# %%
# 
# Exercise 1 — Parameter exploration

# 1. Parameters per encoder stage.
# Deeper stages have more channels (C₁ < C₂ < C₃ < C₄), so even with fewer transformer
# blocks they end up holding most of the encoder's capacity. This matters during
# fine-tuning: freezing stages 3–4 alone already locks the bulk of the model's weights.
for i, stage in enumerate(model.segformer.encoder.block):
    n = sum(p.numel() for p in stage.parameters())
    print(f"Stage {i+1}: {n:,} parameters")

# 2. Decoder breakdown.
# The four projection layers are tiny linear maps (one per encoder stage); the real
# decoder cost lives in the linear_fuse + classifier. Seeing how light the head is
# explains why training the decoder alone is fast and resistant to overfitting.
for i, proj in enumerate(model.decode_head.linear_c):
    n = sum(p.numel() for p in proj.parameters())
    print(f"Decoder projection {i+1}: {n:,} parameters")

clf_params = sum(p.numel() for p in model.decode_head.classifier.parameters())
dec_total = sum(p.numel() for p in model.decode_head.parameters())
print(f"Classifier: {clf_params:,} / {dec_total:,} decoder params "
      f"({100*clf_params/dec_total:.1f}%)")


# %%
# 4 Running a Forward Pass
# Before training, it is useful to trace a dummy forward pass to verify that all shapes are consistent.

# 4.1 Input shape
# Our dataset produces patches of shape (14, H, W) — 14 input channels (12 L2A surface-reflectance bands plus NDVI and NDWI). The DataLoader stacks them into batches (B, 14, H, W). Let us create a random dummy batch:

import torch

B, C, H, W = 2, 14, 512, 512   # batch size, channels, height, width
dummy_input = torch.randn(B, C, H, W)
dummy_labels = torch.randint(0, model.config.num_labels, (B, H, W))

print(f"Input  shape: {tuple(dummy_input.shape)}")
print(f"Labels shape: {tuple(dummy_labels.shape)}")


# %%
# 4.2 Output shapes
# The model has two calling modes depending on whether you pass labels:

model.eval()
with torch.no_grad():
    # Without labels → raw logits at H/4 × W/4
    logits = model(dummy_input)
    print(f"Logits shape (no labels):   {tuple(logits.shape)}")
    # Expected: (2, num_classes, 128, 128)  — quarter resolution

    # With labels → logits upsampled to label resolution
    upsampled = model(dummy_input, dummy_labels)
    print(f"Logits shape (with labels): {tuple(upsampled.shape)}")
    # Expected: (2, num_classes, 512, 512)


# %%
# 4.3 Inspecting intermediate hidden states
# The encoder returns all four intermediate feature maps when 
# output_hidden_states=True. This is exactly how the forward 
# method feeds the decoder:

outputs = model.segformer(
    dummy_input,
    output_hidden_states=True,
    return_dict=True,
)

for i, hs in enumerate(outputs.hidden_states):
    print(f"Stage {i+1} hidden state: {tuple(hs.shape)}")



# %%
# Exercise 2 — Forward pass anatomy
import torch.nn.functional as F

model.eval()
with torch.no_grad():
    # Logits = raw, unbounded per-class scores. They have shape (B, num_classes, H/4, W/4)
    # because the all-MLP decoder fuses everything at the finest encoder scale (H/4).
    logits = model(dummy_input)
    # output_hidden_states=True surfaces the four encoder feature maps. The decoder
    # combines all four — early stages (high resolution) carry local detail, late
    # stages (low resolution) carry semantic context. Returning them is what makes
    # multi-scale fusion possible.
    outputs = model.segformer(
        dummy_input, output_hidden_states=True, return_dict=True
    )

# 1. Manual upsampling.
# Bilinear (not nearest-neighbour) because logits are continuous scores: interpolating
# between two scores is meaningful, whereas a discrete class label is not.
logits_full = F.interpolate(
    logits,
    size=(H, W),
    mode="bilinear",
    align_corners=False,
)
print(f"Manually upsampled: {tuple(logits_full.shape)}")

# 2. Predicted class map.
# softmax converts logits → probabilities; argmax picks the most likely class at each
# pixel. The class axis disappears: (B, num_classes, H, W) → (B, H, W), one int per pixel.
probs = torch.softmax(logits_full, dim=1)
pred  = torch.argmax(probs, dim=1)
print(f"Predicted map shape: {tuple(pred.shape)}")
print(f"Unique predicted classes: {pred.unique().tolist()}")

# 3. Spatial area ratios — stage 1 is at H/4 × W/4 → 1/16 of the input area, and each
# next stage divides that by 4 again (1/64, 1/256, 1/1024).
for i, hs in enumerate(outputs.hidden_states):
    ratio = (hs.shape[-2] * hs.shape[-1]) / (H * W)
    print(f"Stage {i+1}: {ratio:.4f}")


# %%
# 5. Wrapping the Model in a Lightning Module
# PyTorch Lightning separates the model definition from the 
# training logic. The SegmentationModule class in 
# intermediate_solutions/solution_step2/models/module.py wraps 
# SemanticSegmentationSegformer and implements the standard 
# Lightning hooks.
# 5.1 Structure of SegmentationModule
from src.models.module import SegmentationModule
from torch import nn, optim

module = SegmentationModule(
    model=model,
    loss=nn.CrossEntropyLoss(ignore_index=255),
    optimizer=optim.AdamW,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-2},
    scheduler=optim.lr_scheduler.OneCycleLR,
    scheduler_params={},
    scheduler_interval="step",
)


# %%
# 1.2 Load a model from public S3
# For this funathon, a pre-trained segmentation model is publicly available on MinIO — no MLflow credentials or account required. The model artifacts are stored at:

# https://minio.lab.sspcloud.fr/projet-funathon/mlflow-artifacts/
# 1/88138b467a484c54b9935b66460413cd/artifacts/
# ###########
# Exercise 1 bis — Load a pre-trained model from public S3
import s3fs
import mlflow
import requests
import tempfile
import numpy as np
from pathlib import Path

fs = s3fs.S3FileSystem(
    anon=True,
    endpoint_url="https://minio.lab.sspcloud.fr",
)

s3_run_path = "projet-funathon/mlflow-artifacts/1/88138b467a484c54b9935b66460413cd/artifacts/"

s3_model_path = s3_run_path + "model"
local_model_dir = Path(tempfile.mkdtemp()) / "model"

fs.get(s3_model_path, str(local_model_dir), recursive=True)

model = mlflow.pyfunc.load_model(str(local_model_dir))

params_url = "https://minio.lab.sspcloud.fr/" + s3_run_path + "params.json"

response = requests.get(params_url)
run_params = response.json()

n_bands = int(run_params["n_bands"])
tiles_size = int(run_params["tiles_size"])
augment_size = int(run_params["augment_size"])
module_name = run_params["module_name"]
normalization_mean = run_params["normalization_mean"][:n_bands]
normalization_std = run_params["normalization_std"][:n_bands]

print(f"n_bands={n_bands}, tiles_size={tiles_size}, augment_size={augment_size}")
print(f"mean={normalization_mean}")
print(f"std={normalization_std}")




# %%
# Exercise 2 — Run inference on a single Sentinel-2 image
from src.inference.prediction import predict

# image_target = "LU000/2024/4022000_2979190_0_354.tif" # Luxemburg
image_target = "CY000/2024/6432450_1662830_1_3615.tif"  # Nicosia
image_path = (
    "https://minio.lab.sspcloud.fr/projet-funathon/"
    "2026/project3/data/images/" + image_target
)
satellite_img, predictions = predict(
    images=image_path,
    model=model,
    tiles_size=tiles_size,
    augment_size=augment_size,
    n_bands=n_bands,
    normalization_mean=normalization_mean,
    normalization_std=normalization_std,
    module_name=module_name,
)

print(f"Mask shape : {predictions.shape}")
print(f"Classes found : {set(predictions.flatten().tolist())}")

# %%
# 2.2 Display the prediction
# The 10 CLC+ Backbone land-cover classes are mapped to specific 
# colours. A ListedColormap built from these colours ensures the mask 
# and the legend are always consistent.
# The RGB composite is built from bands 4, 3, 2 (Red, Green, Blue), 
# which correspond to 0-based indices 3, 2, 1 in the array. 
# A 98th-percentile normalisation avoids saturation from bright 
# outliers.
# ######################
# Exercise 3 — Display the prediction
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# The 10 CLC+ Backbone classes — same set as defined in 1-acquisition.qmd § Exercise 10.
# Keep the order, IDs and hex colours in sync with that file (and with 5-statistics.qmd
# and map-nuts.qmd) so legends are consistent across the tutorial.
classes = [
    ("Sealed (1)",                        "#FF0100"),
    ("Woody – needle leaved trees (2)",   "#238B23"),
    ("Woody – broadleaved deciduous (3)", "#80FF00"),
    ("Woody – broadleaved evergreen (4)", "#00FF00"),
    ("Low-growing woody plants (5)",      "#804000"),
    ("Permanent herbaceous (6)",          "#CCF24E"),
    ("Periodically herbaceous (7)",       "#FEFF80"),
    ("Lichens and mosses (8)",            "#FF81FF"),
    ("Non- and sparsely-vegetated (9)",   "#BFBFBF"),
    ("Water (10)",                        "#0080FF"),
]

cmap = ListedColormap([color for _, color in classes])
label_to_color = {i + 1: color for i, (_, color) in enumerate(classes)}
legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]

satellite_img_array = satellite_img["array"]
# Pick R/G/B (Sentinel-2 B4/B3/B2 → 0-indexed positions 3/2/1) and move
# channels last so matplotlib can render the natural-colour composite.
rgb = np.transpose(satellite_img_array[[3, 2, 1]], (1, 2, 0)).astype(np.float32)
# Cap brightness at the 98th percentile to avoid a few hot pixels washing out the image.
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(predictions, cmap=cmap, vmin=1, vmax=10)

axes[1].set_title("Predicted land cover")
axes[1].axis("off")

fig.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=True,
)


# %%
# 2.3 Convert predictions to polygons
# create_geojson_from_mask() vectorises the class mask into a 
# GeoDataFrame of polygons. Internally it writes the mask to a temporary 
# GeoTIFF to preserve the georeference, then calls rasterio.features.shapes 
# to extract contiguous regions of identical class. Each row in the 
# output GeoDataFrame has a geometry (polygon) and a label (integer 
# class ID from 1 to 10). Pixels with class value 0 (background / no-data) 
# are automatically excluded.
# ###################
# Exercise 4 — Vectorise the mask and display the polygons
from src.inference.prediction import create_geojson_from_mask

gdf_pred = create_geojson_from_mask(satellite_img, predictions)

print(f"{len(gdf_pred)} polygons extracted")
print(gdf_pred.head())

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB (B4, B3, B2)")
axes[0].axis("off")

axes[1].imshow(predictions, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("Predicted land cover")
axes[1].axis("off")

gdf_pred.plot(
    column="label",
    cmap=cmap,
    vmin=1, vmax=10,
    ax=axes[2],
    legend=False,
)
axes[2].set_title("Predicted polygons")
axes[2].set_aspect("equal")
xmin, ymin, xmax, ymax = gdf_pred.total_bounds
axes[2].set_xlim(xmin, xmax)
axes[2].set_ylim(ymin, ymax)
axes[2].axis("off")

fig.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)



# %%
# 2.4 Display predictions on an interactive map
# Folium renders interactive web maps directly in a Jupyter notebook. 
# The satellite image is overlaid as a semi-transparent raster layer using 
# ImageOverlay, while the predicted polygons are added as a GeoJson layer with 
# per-class colouring.
# Both layers must be in EPSG:4326 (WGS84, latitude/longitude) for Folium to place 
# them correctly. transform_bounds() reprojects the raster bounding box, and 
# gdf_pred.to_crs("EPSG:4326") reprojects the vector polygons.
# Each layer is wrapped in a folium.FeatureGroup so that a LayerControl widget 
# lets the user toggle them on and off independently directly on the map.
# #######################
# Exercise 5 — Display predictions on an interactive Folium map with layer control
import folium
from folium.raster_layers import ImageOverlay
from rasterio.warp import transform_bounds, reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS

# Folium / Leaflet only render WGS84 (EPSG:4326, lat/lon). The Sentinel-2 tile
# lives in EPSG:3035 (metric projection used across Europe), so both the raster
# and the prediction polygons must be reprojected before they reach the map.
src_crs = satellite_img["crs"]
src_transform = satellite_img["transform"]
src_bounds = satellite_img["bounds"]
dst_crs = CRS.from_epsg(4326)

raw_bands = satellite_img_array[[3, 2, 1]].astype(np.float32)
h, w = raw_bands.shape[1], raw_bands.shape[2]
# calculate_default_transform picks an output grid in EPSG:4326 that covers the
# same area; reproject() then resamples each band onto that grid.
dst_transform, dst_w, dst_h = calculate_default_transform(
    src_crs, dst_crs, w, h, *src_bounds
)
rgb_wgs84_bands = np.zeros((3, dst_h, dst_w), dtype=np.float32)
for i in range(3):
    reproject(
        source=raw_bands[i], destination=rgb_wgs84_bands[i],
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,  # bilinear for continuous reflectance values
    )
p98 = np.percentile(rgb_wgs84_bands[rgb_wgs84_bands > 0], 98)
rgb_wgs84 = np.clip(np.transpose(rgb_wgs84_bands, (1, 2, 0)) / p98, 0, 1)
# Reprojection from a tilted source produces empty triangular corners. Build an
# alpha channel from "where was there any data?" so those corners stay transparent.
alpha = (rgb_wgs84_bands.max(axis=0) > 0).astype(np.float32)
rgba_wgs84 = np.dstack([rgb_wgs84, alpha])

west, south, east, north = transform_bounds(src_crs, dst_crs, *src_bounds)
center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

fg_image = folium.FeatureGroup(name="Sentinel-2 RGB", show=True)
ImageOverlay(
    image=rgba_wgs84,
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(fg_image)
fg_image.add_to(m)

fg_pred = folium.FeatureGroup(name="Predicted polygons", show=True)
gdf_pred_wgs84 = gdf_pred.to_crs("EPSG:4326")  # vector polygons reprojected for Folium
folium.GeoJson(
    gdf_pred_wgs84,
    style_function=lambda feature: {
        "fillColor": label_to_color.get(feature["properties"]["label"], "#808080"),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(fg_pred)
fg_pred.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save("Day2_Folium_polygons.html")
m

# %%
# 3 Inference via API
# Running inference locally works well for a single tile, but scaling 
# to an entire NUTS3 region — which may contain dozens of tiles — requires 
# more infrastructure. A REST API has been deployed for this purpose. 
# The API’s interactive documentation is available at 
# https://funathon-2026-project3-api.lab.sspcloud.fr/docs. 
# It lists all available endpoints and lets you test them directly from 
# your browser.
# ###########################
# Exercise 6 - Find the NUTS3 region from a GPS point using the API in Onyxia
import requests

api_url = "https://funathon-2026-project3-api.lab.sspcloud.fr"

# gps_point = [49.63339525016761, 6.1689982433356025]  # [lat, lon] --> Luxemburg
gps_point = [35.1267865, 33.3389614]  # [lat, lon] --> Nicosia

response_nuts = requests.get(
    f"{api_url}/find_nuts",
    params={
        "gps_point": gps_point,
    },
)
response_nuts.raise_for_status()

nuts_id = response_nuts.json()
print(f"NUTS3 region found: {nuts_id}")


# %%
# Exercise 7 - Find a satellite image from a GPS point using the API in Onyxia
import json
import requests

api_url = "https://funathon-2026-project3-api.lab.sspcloud.fr"

# gps_point = [49.63339525016761, 6.1689982433356025]  # [lat, lon] --> Luxemburg
gps_point = [35.1267865, 33.3389614]  # [lat, lon] --> Nicosia
year = 2024

response_find = requests.get(
    f"{api_url}/find_image",
    params={
        "gps_point": gps_point,
        "year": year,
    },
)
response_find.raise_for_status()

image_filepath = response_find.json()
print(f"Image found: {image_filepath}")

