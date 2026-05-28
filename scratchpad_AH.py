# %%
# 1.3 Loading a Sentinel-2 tile
import os
import urllib.request
import rasterio
import numpy as np

tile_url = (
    "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    "project3/data/images/LU000/"
    "2024/4042000_2951690_0_637.tif"
)

with rasterio.open(tile_url) as src:
    tile_crs = src.crs
    tile_bounds = src.bounds
    tile_count = src.count
    tile_height = src.height
    tile_width = src.width
    # Read RGB bands: B4 (Red), B3 (Green), B2 (Blue)
    rgb_data = src.read([4, 3, 2])

print(f"CRS:    {tile_crs}")
print(f"Bounds: {tile_bounds}")
print(f"Shape:  {tile_count} bands x {tile_height} x {tile_width} px")


# %%
# Without clipping 98%
import matplotlib.pyplot as plt

# Transpose to (H, W, 3) and normalize for display
rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
# p98 = np.percentile(rgb, 98)
# rgb = np.clip(rgb / p98, 0, 1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgb)
ax.set_title("Sentinel-2 RGB composite (B4, B3, B2) — LU000")
ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# With clipping 98%
import matplotlib.pyplot as plt

# Transpose to (H, W, 3) and normalize for display
rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
p98 = np.percentile(rgb, 98)
rgb = np.clip(rgb / p98, 0, 1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgb)
ax.set_title("Sentinel-2 RGB composite (B4, B3, B2) — LU000")
ax.axis("off")
plt.tight_layout()
plt.show()


# %%
# Pre-downloaded GeoTiff files:
# https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/images/{NUTS}/{year}/{patch_id}.tif


# %%
# Exercise 2 — Explore raster metadata and create a false-colour composite
import os
import urllib.request
import rasterio
import numpy as np
import matplotlib.pyplot as plt

tile_url = (
    "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    "project3/data/images/LU000/"
    "2024/4042000_2951690_0_637.tif"
)

with rasterio.open(tile_url) as src:
    tile_crs = src.crs
    tile_bounds = src.bounds
    tile_count = src.count
    tile_height = src.height
    tile_width = src.width
    title_profile = src.profile
    # Band Options: B4 (Red), B3 (Green), B2 (Blue), NIR (B8)
    rgb_dataNIR = src.read([8, 3, 2])
    rgb_dataRGB = src.read([4, 3, 2])

print(f"CRS:    {tile_crs}")
print(f"Bounds: {tile_bounds}")
print(f"Shape:  {tile_count} bands x {tile_height} x {tile_width} px")
print(f"Profile: {title_profile}")

rgb_NIR = np.transpose(rgb_dataNIR, (1, 2, 0)).astype(np.float32)
p98_NIR = np.percentile(rgb_NIR, 98)
rgb_NIR = np.clip(rgb_NIR / p98_NIR, 0, 1)

rgb_RGB = np.transpose(rgb_dataRGB, (1, 2, 0)).astype(np.float32)
p98_RGB = np.percentile(rgb_RGB, 98)
rgb_RGB = np.clip(rgb_RGB / p98_RGB, 0, 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# 2. Plot on the first subplot (left)
ax[0].imshow(rgb_NIR)
ax[0].set_title("Sentinel-2 NIR composite (B8, B3, B2) — LU000")
ax[0].axis("off")

# 3. Plot on the second subplot (right)
ax[1].imshow(rgb_RGB)
ax[1].set_title("Sentinel-2 RGB composite (B4, B3, B2) — LU000")
ax[1].axis("off")

plt.tight_layout()
plt.show()

# %%
# Exercise 3 — Explore raster metadata and create a false-colour composite
with rasterio.open(tile_url) as src:
    b8 = src.read(8).astype(np.float32)
    b4 = src.read(4).astype(np.float32)

# ndvi = np.where(b8 + b4 == 0, 0, (b8 - b4) / (b8 + b4))
ndvi = (b8 - b4) / (b8 + b4)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("NDVI — LU000 (2024)")
ax.axis("off")
fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")

plt.tight_layout()
plt.show()


# %%
# 2.3 Finding a NUTS3 region from a city name
import requests
import geopandas as gpd
from shapely.geometry import Point

# Step 1: Geocode the city name
response = requests.get(
    "https://nominatim.openstreetmap.org/search",
    params={"q": "5, rue Alphonse Weicker  Luxembourg", "format": "json", "limit": 1},
    headers={"User-Agent": "funathon-project3"},
)
result = response.json()[0]
lon, lat = float(result["lon"]), float(result["lat"])
print(f"Eurostat coordinates: lon={lon}, lat={lat}")

# Step 2: Create a GeoDataFrame with the point in WGS84, then reproject
city_point = gpd.GeoDataFrame(
    {"city": ["Luxembourg"]}, geometry=[Point(lon, lat)], crs="EPSG:4326" # AH - The Global coordinate system
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
print(f"NUTS3 region: {nuts_code}")  # → LU000


# %%
# Exercise 4 — Geocode a city and build a tile URL
# 
# Step 1: Geocode the city name
import requests
import geopandas as gpd
from shapely.geometry import Point

response = requests.get(
    "https://nominatim.openstreetmap.org/search",
    params={"q": "Odysseos, Strovolos", "format": "json", "limit": 1},
    headers={"User-Agent": "funathon-project3"},
)
result = response.json()[0]
lon, lat = float(result["lon"]), float(result["lat"])
print(f"City: lon={lon}, lat={lat}")

# Step 2: Create a GeoDataFrame with the point in WGS84, then reproject
city_point = gpd.GeoDataFrame(
    {"city": ["Cyprus"]}, geometry=[Point(lon, lat)], crs="EPSG:4326" # AH - The Global coordinate system
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
# # 2.4 Retrieving a tile for a specific city
# import pandas as pd
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt

# # Build the URL to the parquet index
# year = 2024
# nuts_code = "LU000"
# parquet_url = (
#     f"https://minio.lab.sspcloud.fr/projet-funathon/2026/"
#     f"project3/data/images/{nuts_code}/{year}/filename2bbox.parquet"
# )

# # Read the tile index
# tiles = pd.read_parquet(parquet_url)
# print(f"{len(tiles)} tiles in {nuts_code}/{year}")

# # Get city coordinates in EPSG:3035
# x = city_point.geometry.iloc[0].x
# y = city_point.geometry.iloc[0].y
# print(f"City point (EPSG:3035): x={x:.0f}, y={y:.0f}")

# # Find the tile whose bbox contains the city point
# tile_filename = None
# for _, row in tiles.iterrows():
#     xmin, ymin, xmax, ymax = row["bbox"]
#     if xmin <= x <= xmax and ymin <= y <= ymax:
#         tile_filename = row["filename"]
#         break

# print(f"Matching tile: {tile_filename}")

# # Build the full HTTPS URL
# tile_url = (
#     f"https://minio.lab.sspcloud.fr/projet-funathon/2026/"
#     f"project3/data/images/{nuts_code}/{year}/{tile_filename}"
# )

# # Open the tile and display the RGB composite
# with rasterio.open(tile_url) as src:
#     rgb_data = src.read([4, 3, 2])  # Red, Green, Blue bands
#     tile_crs = src.crs
#     tile_bounds = src.bounds

# rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
# rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(rgb)
# ax.set_title(f"Sentinel-2 — {tile_filename}")
# ax.axis("off")
# plt.tight_layout()
# plt.show()


# %%
# Exercise 5 — Find and display the satellite tile for your city
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Build the parquet URL
year = 2024
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
        tile_filename = row["filename"]
        break

print(f"Matching tile: {tile_filename}")

# Step 5: Build the full tile URL
tile_url = (
    f"https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    f"project3/data/images/{nuts_code}/{year}/{tile_filename}"
)

# Step 6: Open, read RGB, normalize and display
with rasterio.open(tile_url) as src:
    rgb_data = src.read([4, 3, 2])
    tile_crs = src.crs
    tile_bounds = src.bounds

print(f"Bounds: {tile_bounds}")

rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(rgb)
ax.set_title(f"Sentinel-2 — {tile_filename}")
ax.axis("off")
plt.tight_layout()
plt.show()



# # %%
# # 3.1 Coordinate Reference Systems
# import geopandas as gpd
# from shapely.geometry import box

# # Create a GeoDataFrame with the tile extent in EPSG:3035
# tile_geom = box(*tile_bounds)
# gdf = gpd.GeoDataFrame({"tile": ["LU000"]}, geometry=[tile_geom], crs="EPSG:3035")

# print("EPSG:3035 bounds:")
# print(gdf.total_bounds)

# # Convert to WGS84 (latitude/longitude)
# gdf_wgs84 = gdf.to_crs("EPSG:4326")
# print("\nEPSG:4326 bounds:")
# print(gdf_wgs84.total_bounds)


# # %%
# # 3.2 Working with GeoDataFrames
# # In practice, you can build a GeoDataFrame directly from rasterio metadata — no filename parsing needed:
# tile_gdf = gpd.GeoDataFrame(
#     {"tile": ["LU000"], "year": [2024]},
#     geometry=[box(*tile_bounds)],
#     crs=tile_crs,
# )
# tile_gdf


# # %%
# # 3.3 Overlaying boundaries on images
# fig, ax = plt.subplots(figsize=(6, 6))
# extent = [tile_bounds.left, tile_bounds.right, tile_bounds.bottom, tile_bounds.top]
# ax.imshow(rgb, extent=extent)
# tile_gdf.boundary.plot(ax=ax, color="red", linewidth=2)
# ax.set_xlabel("Easting (m)")
# ax.set_ylabel("Northing (m)")
# ax.set_title("Sentinel-2 tile with boundary overlay (EPSG:3035)")
# plt.tight_layout()
# plt.show()


# %%
# Exercise 6 — Build a GeoDataFrame from tile bounds and convert CRS
import geopandas as gpd
from shapely.geometry import box

tile_geom = box(*tile_bounds)
gdf = gpd.GeoDataFrame({"tile": [nuts_code]}, geometry=[tile_geom], crs="EPSG:3035")
gdf_wgs84 = gdf.to_crs("EPSG:4326")

print("EPSG:3035 bounds:", gdf.total_bounds)
print("EPSG:4326 bounds:", gdf_wgs84.total_bounds)



# %%
# Exercise 7 — Spatial join with NUTS3 regions
import geopandas as gpd
from shapely.geometry import box

# Step 7a: Load NUTS3 boundaries (EPSG:3035)
nuts_url = (
    "https://gisco-services.ec.europa.eu/distribution/v2/"
    "nuts/geojson/NUTS_RG_01M_2021_3035_LEVL_3.geojson"
)
nuts = gpd.read_file(nuts_url)  # TODO: gpd.read_file(nuts_url)

# Step 7b: Create a GeoDataFrame for the tile
tile_geom = box(*tile_bounds)
tile_gdf = gpd.GeoDataFrame({"tile": [nuts_code]}, geometry=[tile_geom], crs="EPSG:3035")

# Step 7c: Spatial join — find which NUTS3 regions the tile intersects
joined = gpd.sjoin(tile_gdf, nuts, predicate="intersects")  # TODO: gpd.sjoin(tile_gdf, nuts, predicate="intersects")

# Step 7d: Print matching regions
for _, row in joined.iterrows():
    print(f"NUTS_ID: {row['NUTS_ID']}, NUTS_NAME: {row['NUTS_NAME']}")

# Step 7e: Compute tile area in km²
area_km2 = tile_geom.area / 1e6  # TODO: tile_geom.area / 1e6
print(f"Tile area: {area_km2:.2f} km²")



# %%
# Exercise 8 — Display a Sentinel-2 tile on an interactive folium map
# 3.4.1 Static map with matplotlib
# To display the satellite image in geographic coordinates (latitude/longitude), reproject the bounds to WGS84 using rasterio.warp.transform_bounds:

from rasterio.warp import transform_bounds

west, south, east, north = transform_bounds(
    tile_crs, "EPSG:4326", *tile_bounds
)

print(f"WGS84 extent: W={west:.4f}, S={south:.4f}, E={east:.4f}, N={north:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(rgb, extent=[west, east, south, north])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Sentinel-2 tile in WGS84 coordinates")
plt.tight_layout()
plt.show()

# 3.4.2 Interactive map with folium
# For an interactive web map, use Folium to overlay the satellite image on a basemap:
import folium
from folium.raster_layers import ImageOverlay

center_lat = (south + north) / 2
center_lon = (west + east) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

ImageOverlay(
    image=rgb,
    bounds=[[south, west], [north, east]],
    opacity=0.8,
).add_to(m)

m


# %%
# 4.3 Loading and displaying a label
import urllib.request
import io
import numpy as np

# Label URL for a LU000 patch, year 2021
label_url = (
    "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    "project3/data/labels/LU000/"
    "2021/4042000_2951690_0_637.npy"
)

with urllib.request.urlopen(label_url) as response:
    label_array = np.load(io.BytesIO(response.read()))

print(f"Label shape: {label_array.shape}")
print(f"Data type:   {label_array.dtype}")
print(f"Classes:     {np.unique(label_array)}")


# %%
# 4.3 continued...
# We can visualise the label using a custom colormap that assigns a colour to each of the 10 land-cover classes:
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# CLC+ class names and colours
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
cmap = ListedColormap([color for _, color in classes])

# Load the matching satellite image
image_url = (
    "https://minio.lab.sspcloud.fr/projet-funathon/2026/"
    "project3/data/images/LU000/"
    "2021/4042000_2951690_0_637.tif"
)
with rasterio.open(image_url) as src:
    rgb_data = src.read([4, 3, 2])

rgb = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

# Side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(rgb)
axes[0].set_title("Sentinel-2 RGB")
axes[0].axis("off")

axes[1].imshow(label_array, cmap=cmap, vmin=1, vmax=10)
axes[1].set_title("CLC+ Backbone label")
axes[1].axis("off")

legend_elements = [
    Patch(facecolor=color, edgecolor="black", label=label)
    for label, color in classes
]
fig.legend(
    handles=legend_elements,
    loc="center right",
    bbox_to_anchor=(1.30, 0.5),
    frameon=True,
)
plt.tight_layout()
plt.show()


# %%
# Exercise 9 — Load a CLC+ label from S3
import urllib.request
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

nuts_code = "CY000"
year = 2021
patch_id = "6432450_1665330_1_3521"

label_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/labels/{nuts_code}/{year}/{patch_id}.npy"

with urllib.request.urlopen(label_url) as response:
    my_label = np.load(io.BytesIO(response.read()))

print(f"Shape: {my_label.shape}")
print(f"Classes: {np.unique(my_label)}")

cmap = ListedColormap(
    [
        "#FF0100",
        "#238B23",
        "#80FF00",
        "#00FF00",
        "#804000",
        "#CCF24E",
        "#FEFF80",
        "#FF81FF",
        "#BFBFBF",
        "#0080FF",
    ]
)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(my_label, cmap=cmap, vmin=1, vmax=10)
ax.set_title(f"CLC+ label — {nuts_code}/{year}/{patch_id}")
ax.axis("off")
plt.show()


# %%
# Exercise 10 — Overlay the satellite image and CLC+ label on an interactive map
import numpy as np
import urllib.request
import io
import rasterio
import folium
from rasterio.warp import transform_bounds
from matplotlib.colors import to_rgba

nuts_code = "CY000"
year = 2021
patch_id = "6432450_1665330_1_3521"

label_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/labels/{nuts_code}/{year}/{patch_id}.npy"
image_url = f"https://minio.lab.sspcloud.fr/projet-funathon/2026/project3/data/images/{nuts_code}/{year}/{patch_id}.tif"

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

print(label_url)
print(image_url)

# Step 1: Load satellite image
with rasterio.open(image_url) as src:
    rgb_data = src.read([4, 3, 2])
    bounds_3035 = src.bounds
    crs = src.crs


rgb_overlay = np.transpose(rgb_data, (1, 2, 0)).astype(np.float32)
rgb_overlay = np.clip(rgb_overlay / np.percentile(rgb_overlay, 98), 0, 1)

# Step 2: Load the matching label
with urllib.request.urlopen(label_url) as response:
    label = np.load(io.BytesIO(response.read()))

# Step 3: Convert label to RGBA
color_lut = np.zeros((11, 4), dtype=np.float32)
color_lut[0] = [0, 0, 0, 0]
for i, (_, hex_color) in enumerate(classes, start=1):
    color_lut[i] = list(to_rgba(hex_color, alpha=0.7))

label_rgba = color_lut[label]

# Step 4: Reproject bounds to WGS84
west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds_3035)

center_lat = (south + north) / 2
center_lon = (west + east) / 2

# Step 5: Create the map
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Step 6: Add overlays
folium.raster_layers.ImageOverlay(
    image=rgb_overlay,
    bounds=[[south, west], [north, east]],
    name="Sentinel-2 RGB",
).add_to(m)

folium.raster_layers.ImageOverlay(
    image=label_rgba,
    bounds=[[south, west], [north, east]],
    name="CLC+ Label",
    opacity=0.8,
).add_to(m)

# Step 7: Layer control
folium.LayerControl().add_to(m)
m
