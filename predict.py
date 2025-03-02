import torch
import timm
import base64
import io
import os
import numpy as np
from PIL import Image
from torchvision import transforms

# Define configurations
class Config:
    backbone = "resnet18"
    n_classes = 8
    image_size = 200
    device = "cpu"  # Use CPU for inference
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

cfg = Config()

# Class names
class_names = [
    "1 polyethylene (PET)",
    "2 high density polyethylene (HDPE/PEHD)",
    "3 polyvinyl chloride (PVC)",
    "4 low density polyethylene (LDPE)",
    "5 polypropylene (PP)",
    "6 polystyrene (PS)",
    "7 other resins",
    "8 no plastic"
]

# Load model (lazy loading)
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "model", "model_fold_0.pth")
        model = timm.create_model(cfg.backbone, pretrained=False, num_classes=cfg.n_classes)
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.to(cfg.device)
        model.eval()
    return model

def preprocess_image(image_data):
    """Preprocess the image to match the model's expected input"""
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])

    # Convert PIL image
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(cfg.device)

def predict_image_base64(image_base64):
    """Predict from base64 encoded image"""
    try:
        # Remove data URL prefix if present
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,")[1]
            
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Load model
        model = get_model()
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item() * 100
        
        # Get predicted class name
        predicted_class = class_names[predicted_class_idx]
        
        # Return result
        return {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")