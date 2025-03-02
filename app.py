import os
import logging
import uvicorn
import base64
import io
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from predict import predict_image_base64

# Create FastAPI app
app = FastAPI(title="Plastic Waste Classifier API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (optional - for testing)
app.mount("/static", StaticFiles(directory="static"), name="static")

# API endpoint for predictions
@app.post("/api/predict")
async def predict(request: Request):
    try:
        # Parse request body
        data = await request.json()
        image_base64 = data.get("image")
        
        if not image_base64:
            return {"error": "No image data provided"}
        
        # Make prediction
        result = predict_image_base64(image_base64)
        return result
        
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return {"error": str(e)}

# Simple test endpoint
@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# For local testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)