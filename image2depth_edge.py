import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

# set the direction location
input_dir = 'input'
output_dir = 'output'
output_a_dir = os.path.join(output_dir, 'A')
output_b_dir = os.path.join(output_dir, 'B')
os.makedirs(output_a_dir, exist_ok=True)
os.makedirs(output_b_dir, exist_ok=True)

# MiDaS model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
model.eval()

# Resize the image to form we needed 
transform = Compose([Resize((384, 384), interpolation=InterpolationMode.BILINEAR),ToTensor()])

def process_image(image_path, output_a_path, output_b_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge detected
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite(output_a_path, edges)

    # transform image to PIL for torchvision.transforms
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # depth estimate
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(img_tensor)
    depth = depth.squeeze().cpu().numpy()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype('uint8')
    cv2.imwrite(output_b_path, depth_normalized)

#Get the image name
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in tqdm(image_files, desc="Processing images", unit="image"):
    input_path = os.path.join(input_dir, filename)
    output_a_path = os.path.join(output_a_dir, filename)
    output_b_path = os.path.join(output_b_dir, filename)
    
    process_image(input_path, output_a_path, output_b_path)

print("Finished")


