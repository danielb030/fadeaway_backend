from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image

import uvicorn
import os
import base64
import io
import requests
import numpy as np
import cv2

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

def detect_and_crop_head_from_array(cv_image, factor=1.7):
    # Load the pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=10)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        center_x = x + w // 2
        center_y = y + h // 2
        size = int(max(w, h) * factor)
        x_new = max(0, center_x - size // 2)
        y_new = max(0, center_y - size // 2)
        cropped_head = cv_image[y_new:y_new+size, x_new:x_new+size]
        return cropped_head
    else:
        return cv_image
    
@app.post("/process-image/")
async def process_images(request: Request):
    data = await request.json()
    image_b64 = data.get("image")
    if not image_b64:
        return {"error": "No image data provided"}

    # Decode base64 image
    input_data = base64.b64decode(image_b64.split(",")[-1])
    # Process image (simulate your remove function)
    # output_data = remove(image_data)  # Uncomment if you have a remove() function
    output_data = remove(input_data)  # Replace with actual processing

    # Convert result to OpenCV format
    image = Image.open(io.BytesIO(output_data)).convert("RGBA")
    open_cv_image = np.array(image)
    head = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)

    cropped_head = detect_and_crop_head_from_array(head)
    if cropped_head is None:
        return {"error": "No faces detected in the input image."}

    # Save processed image
    output_path = os.path.join("uploads", "processed_image.png")
    cv2.imwrite(output_path, cropped_head)

    return FileResponse(output_path, media_type="image/png", filename="processed_image.png")

@app.post("/upload/")
async def upload_images(request: Request):
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
async def show_progress(request: Request):
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

@app.get("/")
def root():
    return {"message": "It works!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render needs this!
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")