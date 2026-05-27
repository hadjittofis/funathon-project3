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



# %%
# 3.1 Coordinate Reference Systems
import geopandas as gpd
from shapely.geometry import box

# Create a GeoDataFrame with the tile extent in EPSG:3035
tile_geom = box(*tile_bounds)
gdf = gpd.GeoDataFrame({"tile": ["LU000"]}, geometry=[tile_geom], crs="EPSG:3035")

print("EPSG:3035 bounds:")
print(gdf.total_bounds)

# Convert to WGS84 (latitude/longitude)
gdf_wgs84 = gdf.to_crs("EPSG:4326")
print("\nEPSG:4326 bounds:")
print(gdf_wgs84.total_bounds)


# %%
# 3.2 Working with GeoDataFrames
# In practice, you can build a GeoDataFrame directly from rasterio metadata — no filename parsing needed:
tile_gdf = gpd.GeoDataFrame(
    {"tile": ["LU000"], "year": [2024]},
    geometry=[box(*tile_bounds)],
    crs=tile_crs,
)
tile_gdf


# %%
# 3.3 Overlaying boundaries on images
fig, ax = plt.subplots(figsize=(6, 6))
extent = [tile_bounds.left, tile_bounds.right, tile_bounds.bottom, tile_bounds.top]
ax.imshow(rgb, extent=extent)
tile_gdf.boundary.plot(ax=ax, color="red", linewidth=2)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_title("Sentinel-2 tile with boundary overlay (EPSG:3035)")
plt.tight_layout()
plt.show()


# %%
# Exercise 6 — Build a GeoDataFrame from tile bounds and convert CRS

