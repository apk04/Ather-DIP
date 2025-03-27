# Ather Scooter Detection using YOLOv5
This project uses a YOLOv5 model to detect scooters in images and verify if they are Ather scooters based on logo matching.
## Installation
Ensure you have Python 3.x installed. Then, install the required dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pillow
## Project Structure
├── images/            # Folder containing test images
├── yolov5s.pt         # YOLOv5 model weights
├── atherrr.py    # Python script for detection
├── Ather_New_Logo.jpg # Reference image for logo matching
├── README.md          # Project documentation
## Usage
Run the following command to test an image:
```bash
python atherrr.py --image_path images/sample.jpg
The script will output whether an Ather scooter is detected in the given image.
- Ensure that `yolov5s.pt` is in the project directory.
- If `torch.hub.load` fails, try running YOLO manually from the `ultralytics/yolov5` repo.
