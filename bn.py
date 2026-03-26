import ray
import numpy as np
import rasterio

@ray.remote
def process_tile(red, nir):
    denom = red + nir
    # Calculate NDVI safely
    ndvi = np.divide((nir - red), denom, out=np.zeros_like(red), where=denom!=0)
    return np.nanmean(ndvi)

def compute_and_save_ndvi(red_path, nir_path):
    TILE_SIZE = 2048 # Larger tiles = fewer tasks = less overhead
    all_means = []
    
    with rasterio.open(red_path) as red_ds, rasterio.open(nir_path) as nir_ds:
        print(f" Processing 24k image in serial-batch mode...")

        for i in range(0, red_ds.height, TILE_SIZE):
            for j in range(0, red_ds.width, TILE_SIZE):
                window = rasterio.windows.Window(j, i, TILE_SIZE, TILE_SIZE)
                
                # Read only what we need
                r = red_ds.read(1, window=window).astype(np.float32)
                n = nir_ds.read(1, window=window).astype(np.float32)
                
                # Execute and WAIT immediately (ray.get) to keep RAM low
                result = ray.get(process_tile.remote(r, n))
                all_means.append(result)
                
                # Periodic progress update for the terminal
                if len(all_means) % 10 == 0:
                    print(f"Progress: {len(all_means)} blocks calculated...")

    return {"mean_ndvi": float(np.mean(all_means)), "output_file": "ndvi_final.tif"}