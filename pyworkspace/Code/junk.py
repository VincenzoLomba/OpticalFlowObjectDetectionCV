
import logger as log
import os
from opticalFlow import loadVideoFrames
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

p = Path(r"C:\Users\vince\OneDrive\Documenti\Università\Magistrale\Second Year\Topic Highlights\Final Project\Materials\EgoMotionSubtraction\EgoMotionOFCode\data\test_unreal1\image\1.jpg")
long_path = Path(p).absolute().resolve()
with Image.open(long_path) as pil_img:
    c = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
# c = cv2.imread(str(long_path))
print(c)

# Loading video frames
print(os.sep)
inputVideoPath = "C:\\Users\\vince\\Downloads\\rgb_output.avi"  # Replace with your video path
log.setActive("LOADER")
frames, fps, width, height = loadVideoFrames(inputVideoPath)
if not frames: exit()

vels = np.load("C:\\Users\\vince\\Downloads\\camera_velocities.npy")
print(vels.shape)
linear_x = vels[:, 0]
linear_y = vels[:, 1]
linear_z = vels[:, 2]
angular_x = vels[:, 3]
angular_y = vels[:, 4]
angular_z = vels[:, 5]
print("linear_x", linear_x.shape)
print("linear_y", linear_y.shape)
print("linear_z", linear_z.shape)
print("angular_x", angular_x.shape)
print("angular_y", angular_y.shape)
print("angular_z", angular_z.shape)

depth = np.load("C:\\Users\\vince\\Downloads\\depth_tensor.npy")
print(depth.shape)

print(f"Loaded {len(frames)} frames with resolution {width}x{height} at {fps} FPS.")
