import uvicorn
import os
import ray
from fastapi import FastAPI
from pydantic import BaseModel
from bn import compute_and_save_ndvi

# FIX: Force Ray to be more patient on Windows hardware
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "60"
os.environ["RAY_CHORD_MAX_RETRIES"] = "3"

app = FastAPI()

class AnalysisRequest(BaseModel):
    red_path: str
    nir_path: str

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    print(f"Received Request: {request.red_path}")
    # Ray logic
    result = compute_and_save_ndvi(request.red_path, request.nir_path)
    return {
        "status": "Success",
        "average_ndvi": result["mean_ndvi"],
        "file": result["output_file"]
    }

if __name__ == "__main__":
    print("RAY SERVER: Starting on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, timeout_keep_alive=300)