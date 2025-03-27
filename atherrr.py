import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load a pretrained YOLO model (or use a custom-trained model for Ather detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLO model's input size
    transforms.ToTensor(),
])

def detect_ather_scooter(image_path):
    # Load image
    image = Image.open(image_path)
    results = model(image)  # Run YOLO model on the image
    
    # Convert results to pandas dataframe
    df = results.pandas().xyxy[0]

    # Check for class labels related to 'scooter' or 'motorcycle'
    detected_scooters = df[df['name'].isin(['motorcycle', 'scooter'])]

    if not detected_scooters.empty:
        print("Scooter detected! Checking if it's an Ather...")
        
        # Additional logic to verify Ather scooter using color, shape, etc.
        image_cv = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Check Ather’s distinctive color or logo using feature matching (example)
        ather_logo = cv2.imread('Ather_New_Logo.jpg', 0)  # Load Ather logo in grayscale
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ather_logo, None)
        kp2, des2 = sift.detectAndCompute(image_gray, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test for matching
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) > 10:  # Threshold for Ather logo match
            print("✅ Ather Scooter Identified!")
            return True
        else:
            print("❌ Not an Ather Scooter.")
            return False
    else:
        print("No scooter detected.")
        return False

# Test with an image
image_path = "Untitled-design-2020-01-29T150530.589.jpg"
detect_ather_scooter(image_path)
