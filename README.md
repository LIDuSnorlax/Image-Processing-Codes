# Image to edge and depth map
Batch process images to get edge and depth maps.
Use Opencv and MiDaS model to batch process images to get canny edge and depth map of the image.

MiDas: https://github.com/isl-org/MiDaS

Once configured the environment, just run the script, here is an example:
![example](https://github.com/user-attachments/assets/1c08673a-7497-4f06-a2bb-0ae5249a4cfb)
Image source:  17flowers https://www.robots.ox.ac.uk/~vgg/data/flowers/17/


## Environment Configuration

    pip install opencv-python torch torchvision pillow tqdm
    
    conda install -c conda-forge opencv pytorch torchvision pillow tqdm
