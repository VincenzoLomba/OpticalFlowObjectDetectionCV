
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

"""
import numpy as np

def estimate_v(flow, G):
    h, w, _ = flow.shape
    print("ALTEZZA", h)
    print("LARGHEZZA", w)
    print(G.shape)
    v_estimatesx = []
    v_estimatesy = []
    v_estimatesz = []
    
    ungood = 0
    load = 0
    for i in range(h):
        for j in range(w):
            load += 1
            # Estrai G_{i,j} (3x2) e flow_{i,j} (2,)
            #G_ij = G[i, j]  # Shape (3, 2)
            flow_ij = flow[i, j]  # Shape (2,)
            
            # Calcola v per questo pixel: v = (G^T G)^{-1} G^T flow
            #G_T = G_ij.T  # Shape (2, 3)
            #G_T_G = np.dot(G_T, G_ij)  # Shape (3, 3)
            #G_T_flow = np.dot(G_T, flow_ij)  # Shape (3,)
            
            try:
                # Risolvi il sistema lineare
                #v_ij = np.linalg.solve(G_T_G, G_T_flow)
                G_ij = np.reshape(G[i, j], (2,3))# Shape (3, 2)
                flow_ij = np.reshape(flow[i, j], (2,1)) # Shape (2,)
                v_ij = np.linalg.pinv(G_ij)@flow_ij  # Shape (3,)
                v_estimatesx.append(v_ij[0])
                v_estimatesy.append(v_ij[1])
                v_estimatesz.append(v_ij[2])
            except np.linalg.LinAlgError:
                # Se G_T_G è singolare, salta questo pixel
                ungood += 1
    
    # Calcola la media di tutte le stime di v
    #if len(v_estimates) == 0:
    #    raise ValueError("Nessuna stima valida per v (matrice singolare ovunque)")
    #print(f"Ungood pixels AAA: {ungood}")
    #print(f"Uhgood pixels EEE: {len(v_estimates)}")
    #print(f"Load: {load}")
    #v_mean = np.mean(v_estimates, axis=0)
    #return v_mean.reshape(3, 1)  # Ritorna v come (3, 1)
    return v_estimatesx, v_estimatesy, v_estimatesz
    """

"""
def loadCameraVelocitiesFromStationaryEnvironment(
        coloredFrames: List[np.ndarray],
        focalLength: float,
        videoDepths: List[np.ndarray],
        velocitiesSelection: List[bool]):
    
    framesWidth = coloredFrames[0].shape[1]
    framesHeight = coloredFrames[0].shape[0]

    # Unpacking video depths data and checking for consistency
    videoDepthsFramesQuantity = len(videoDepths)
    videoDepthsWidth = len(videoDepths[0][0])
    videoDepthsHeight = len(videoDepths[0])
    if videoDepthsFramesQuantity != len(coloredFrames) or videoDepthsWidth != framesWidth or videoDepthsHeight != framesHeight:
        log.error("Error: the provided video depths data is inconsistent with the video frames!")

    if coloredFrames is None: log.error("Error: the provided colored frames data is a None object!")
    grayFrames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in coloredFrames]

    G,H = computeEgoMotionMatrices(framesWidth, framesHeight, focalLength)
    w = []
    for j in range(130, len(grayFrames)):
        prevFrame = grayFrames[j - 1]
        currentFrame = grayFrames[j]
        flow = computeOpticalFlow(prevFrame, currentFrame)
        print(flow.shape)
        vsx = []
        vsy = []
        for ii in range(0, len(flow)): # scorrimento h
            for jj in range(0, len(flow[0])): # scorrimento w
                pVel = np.reshape(flow[ii][jj], (2, 1))
                vsx.append(pVel[0])
                vsy.append(pVel[1])
                # print(pVel)
        from miscellaneous import estimate_v, histogram, plot
        print("> ", len(vsx))
        print("> ", len(vsy))
        print("means: ", np.mean(vsx), " - ", np.mean(vsy))
        print("maxims: ", np.max(vsx), " - ", np.max(vsy))
        print("minims: ", np.min(vsx), " - ", np.min(vsy))
        print("stds: ", np.std(vsx), " - ", np.std(vsy))
        plot(vsx, "X Velocity")
        plot(vsy, "Y Velocity")
        #histogram(vsx)
        #histogram(vsy)
        #pVelsx, pVelsy, pVelsz = estimate_v(flow, G)
        #plot(pVelsx, "X Velocity")
        #plot(pVelsy, "Y Velocity")
        #plot(pVelsz, "Z Velocity")
        print(flow.shape)
        print("> ", len(flow)," - ", len(flow[0]))
        print("Total pixels: ", len(flow[0]) * len(flow[0][0]))
        # histogram(pVels[0])
        # """