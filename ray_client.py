import ray
import numpy as np
import rasterio

if not ray.is_initialized():
    ray.init(num_cpus=4, ignore_reinit_error=True)

@ray.remote
def process_tile(red, nir):
    denom = red + nir
    # NDVI where denominator isn't zero
    ndvi = np.divide((nir - red), denom, out=np.zeros_like(red), where=denom!=0)
    return np.nanmean(ndvi)

def compute_and_save_ndvi(red_path, nir_path):
    TILE_SIZE = 2024 
    all_means = []
    
    with rasterio.open(red_path) as red_ds, rasterio.open(nir_path) as nir_ds:
        futures = []
        print(f"Processing 24k image in batches...")

        for i in range(0, red_ds.height, TILE_SIZE):
            for j in range(0, red_ds.width, TILE_SIZE):
                window = rasterio.windows.Window(j, i, TILE_SIZE, TILE_SIZE)
                r = red_ds.read(1, window=window).astype(np.float32)
                n = nir_ds.read(1, window=window).astype(np.float32)
                
                futures.append(process_tile.remote(r, n))

                # Process every 50 tiles to prevent RAM spikes
                if len(futures) >= 50:
                    batch_results = ray.get(futures)
                    all_means.extend(batch_results)
                    futures = [] # Clear the "waiting" list
                    print(f"Batch complete. Tiles processed: {len(all_means)}")

        # Catch any remaining tiles
        if futures:
            all_means.extend(ray.get(futures))

    final_avg = np.mean(all_means)
    return {"mean_ndvi": float(final_avg), "output_file": "ndvi_final.tif"}