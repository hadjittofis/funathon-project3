# %%
# [A] Import necessary packages and input options
import numpy as np
import urllib.request
import io
import rasterio
import folium
import requests
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

from rasterio.warp import transform_bounds, calculate_default_transform, Resampling, reproject
from rasterio.crs import CRS
from matplotlib.colors import to_rgba
from shapely.geometry import Point

# Label coloring classes
classes = [
    ("Sealed (1)", "#FF0100"),
    ("Woody -- needle leaved trees (2)", "#238B23"),
    ("Woody -- Broadleaved deciduous trees (3)", "#80FF00"),
    ("Woody -- Broadleaved evergreen trees (4)", "#00FF00"),
    ("Low-growing woody plants (bushes, shrubs) (5)", "#804000"),
    ("Permanent herbaceous (6)", "#CCF24E"),
    ("Periodically herbaceous (7)", "#FEFF80"),
    ("Lichens and mosses (8)", "#FF81FF"),
    ("Non- and sparsely-vegetated (9)", "#BFBFBF"),
    ("Water (10)", "#0080FF"),
]

year = 2021
address = "Kountourioti, Strovolos"
city = "Nicosia"

# Reproject bounds to WGS84
reproject_toworld = True



# %%
# [B] Find lon, lat and "nuts_code" for address above
response = requests.get(
    "https://nominatim.openstreetmap.org/search",
    params={"q": address, "format": "json", "limit": 1},
    headers={"User-Agent": "funathon-project3"},
)
result = response.json()[0]
lon, lat = float(result["lon"]), float(result["lat"])
print(f"City: lon={lon}, lat={lat}")

# Step 2: Create a GeoDataFrame with the point in WGS84, then reproject
city_point = gpd.GeoDataFrame(
    {"city": [city]}, geometry=[Point(lon, lat)], crs="EPSG:4326" # AH - The Global coordinate system
)
city_point = city_point.to_crs("EPSG:3035") # AH - Convert to the European system (to which NUTS corresponds to)
print(f"City point: {city_point}")

# Step 3: Load NUTS3 boundaries and spatial join
nuts_url = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "nuts/geojson/NUTS_RG_01M_2021_3035_LEVL_3.geojson"
)
nuts = gpd.read_file(nuts_url)
city_nuts = gpd.sjoin(city_point, nuts, predicate="within")
nuts_code = city_nuts.iloc[0]["NUTS_ID"]
print(f"NUTS3 region: {nuts_code}") 

base_url = f"s3://projet-funathon/2026/project3/data/images/{nuts_code}"
print(base_url)




# %%
# [C] Find relevant geo tile "tile_filename"
# Step 1: Build the parquet URL
parquet_url = (
    f"https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    f"project3/data/images/{nuts_code}/{year}/filename2bbox.parquet"
)

# Step 2: Read the tile index
tiles = pd.read_parquet(parquet_url)
print(f"{len(tiles)} tiles available")

# Step 3: Get city coordinates in EPSG:3035
x = city_point.geometry.iloc[0].x
y = city_point.geometry.iloc[0].y

# Step 4: Find the matching tile
tile_filename = None
for _, row in tiles.iterrows():
    xmin, ymin, xmax, ymax = row["bbox"]
    if xmin <= x <= xmax and ymin <= y <= ymax:
        tile_filename = row["filename"].removesuffix(".tif")
        break

print(f"Matching tile: {tile_filename}")

# Step 5: Build the full tile and label URLs
label_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/labels/{nuts_code}/{year}/{tile_filename}.npy"
image_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/images/{nuts_code}/{year}/{tile_filename}.tif"



# %%
# [D] Make image ready for displaying
with rasterio.open(image_url) as src:
    rgb_data = src.read([4, 3, 2])
    tile_crs = src.crs
    tile_bounds = src.bounds
    # Extras for reproject_toworld = True. If set to False, images do not 
    # align in interactive folium map
    title_transform = src.transform
    title_width = src.width
    title_height = src.height
    title_raw_bands = src.read([4, 3, 2]).astype(np.float32)

if reproject_toworld:
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        tile_crs, dst_crs, title_width, title_height, *tile_bounds
    )
    rgb_wgs84 = np.zeros((3, dst_h, dst_w), dtype=np.float32)
    for i in range(3):
        reproject(
            source=title_raw_bands[i],
            destination=rgb_wgs84[i],
            src_transform=title_transform,
            src_crs=tile_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    alpha = (rgb_wgs84.max(axis=0) > 0).astype(np.float32)
    rgba = np.dstack([np.transpose(rgb_wgs84, (1, 2, 0)), alpha])
else:
    rgba = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
    rgba = np.clip(rgba / np.percentile(rgba, 98), 0, 1)


# %%
# [E] Make label ready for displaying
with urllib.request.urlopen(label_url) as response:
    label = np.load(io.BytesIO(response.read()))

if reproject_toworld:
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        tile_crs, dst_crs, title_width, title_height, *tile_bounds
    )
    
    # Initialize destination array for the label (2D array matching dst_h and dst_w)
    # Default to 0 (background/unclassified)
    label_wgs84 = np.zeros((dst_h, dst_w), dtype=label.dtype)
    
    # Reproject using NEAREST neighbor to preserve class IDs
    reproject(
        source=label,
        destination=label_wgs84,
        src_transform=title_transform,
        src_crs=tile_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest, # CRITICAL for classification labels
    )
    
    final_label = label_wgs84
else:
    final_label = label

# Convert label to RGBA
color_lut = np.zeros((11, 4), dtype=np.float32)
color_lut[0] = [0, 0, 0, 0]
for i, (_, hex_color) in enumerate(classes, start=1):
    color_lut[i] = list(to_rgba(hex_color, alpha=0.7))

label_rgba = color_lut[final_label]





# %%
# [F] Display Image and Label in two different graphs
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgba)
ax.set_title(f"Sentinel-2 — {tile_filename}")
ax.axis("off")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(label_rgba)
ax.set_title(f"Label — {tile_filename}")
ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# [G] Display Image and Label in the SAME graph, side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(rgba)
ax[0].set_title(f"Sentinel-2 — {tile_filename}")
ax[0].axis("off")

ax[1].imshow(label_rgba)
ax[1].set_title(f"Label — {tile_filename}")
ax[1].axis("off")

plt.tight_layout()
plt.show()


# %%
# [H] Create interactive graph and overlay on top the image and the label

# Reproject bounds to WGS84
west, south, east, north = transform_bounds(tile_crs, "EPSG:4326", *tile_bounds)

center_lat = (south + north) / 2
center_lon = (west + east) / 2

# Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# Add overlays
folium.raster_layers.ImageOverlay(
    image=rgba,
    bounds=[[south, west], [north, east]],
    name="Sentinel-2 RGB",
    opacity=0.9,
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=label_rgba,
    bounds=[[south, west], [north, east]],
    name="CLC+ Label",
    opacity=0.8,
).add_to(m)

# Layer control
folium.LayerControl().add_to(m)
m.save("Day1_Folium.html") # Cannot display in Onyxia, need to save and open it
m


