from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import os
import base64
import io

app = FastAPI()

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

@app.post("/upload/")
async def process_images(request: Request):
    data = await request.json()
    images = data.get("images", [])
    saved_files = []

    for idx, img_str in enumerate(images):
        # Remove header if present
        if "," in img_str:
            img_str = img_str.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        img = Image.open(io.BytesIO(img_bytes))
        file_path = f"uploads/uploaded_{idx}.png"
        img.save(file_path)
        saved_files.append(file_path)

    # Here you would generate your .obj file using the saved images
    # For demo, just use a static .obj file
    output_path = "models/head_model1.obj"
    # ... generate or copy your .obj file to output_path ...

    return FileResponse(
        output_path,
        media_type="text/plain",  # or "model/obj"
        filename="head_model1.obj"
    )