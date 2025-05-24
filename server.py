from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import os
import base64
import io
import requests
import time
import zipfile

app = FastAPI()
mesh_api_key = "msy_qOnp1jEDQ39zjTCLIjqGfvBrZuAFCHbo3Hec"
# CORS middleware to allow requests from your frontend
# Add this block before your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fadeaway-frontend.vercel.app",  # your deployed frontend
        "http://localhost:8080",                 # for local dev (optional)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure a folder for uploads exists
os.makedirs("uploads", exist_ok=True)

YOUR_API_KEY = mesh_api_key

headers = {
    "Authorization": f"Bearer {YOUR_API_KEY}"
}

@app.post("/upload/")
async def process_images(request: Request):
    data = await request.json()
    imageUrls = data.get("imageUrls", [])

    payload = {
        # Using data URI example
        # image_urls: [
        #   f'data:image/png;base64,{YOUR_BASE64_ENCODED_IMAGE_DATA_1}',
        #   f'data:image/png;base64,{YOUR_BASE64_ENCODED_IMAGE_DATA_2}'
        # ],
        "image_urls": imageUrls,
        "should_remesh": True,
        "should_texture": True,
        "target_polycount": 250000,
        # "texture_prompt": "Since the purpose is to create a human model, the head should be made especially well. Or, just the head is fine. Make the hair very delicate. This is the model that will be used for the barber.",
        "moderation": True
    }

    response = requests.post(
        "https://api.meshy.ai/openapi/v1/multi-image-to-3d",
        headers=headers,
        json=payload,
    )
    response.raise_for_status()
    return (response.json().get("result"))


@app.post("/progress/")
async def process_model(request: Request):
    data = await request.json()
    task_id = data.get("task_id")

    status = None
    result = None
    poll_url = f"https://api.meshy.ai/openapi/v1/multi-image-to-3d/{task_id}"
    poll_response = requests.get(poll_url, headers=headers)
    poll_response.raise_for_status()
    result = poll_response.json()
    progress = result.get("progress")
    status = result.get("status")
    if status != "SUCCEEDED":
        return progress
    else:
        return result

@app.post("/download-model/")
async def download_model(request: Request):
    data = await request.json()
    model_url = data.get("model_url")
    print(model_url)
    if not model_url:
        return {"error": "No model URL found"}

    # Download the model file
    response = requests.get(model_url)
    response.raise_for_status()

    # Determine file extension (default to .glb)
    ext = os.path.splitext(model_url.split("?")[0])[1] or ".glb"
    filename = f"model{ext}"
    file_path = os.path.join("uploads", filename)
    with open(file_path, "wb") as f:
        f.write(response.content)

    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)