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
    {"city": ["Luxembourg"]}, geometry=[Point(lon, lat)], crs="EPSG:4326" # The Global coordinate system
)
city_point = city_point.to_crs("EPSG:3035") # To European system that NUTS corresponds to
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
