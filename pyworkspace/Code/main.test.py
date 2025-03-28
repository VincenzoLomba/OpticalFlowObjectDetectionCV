
from pathlib import Path
import cv2, os
import logger as log
import numpy as np
from opticalFlow import loadImagesAsFrames, computeOpticalFlows, saveFramesToVideo
import matplotlib.pyplot as plt

def plot(vector, title):
    plt.figure(figsize=(10, 5))
    plt.plot(vector, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('time')
    plt.grid(True)
    plt.show()

def resample(array, numberOfSamples):
    l = array.shape[0]
    if l == numberOfSamples: return array
    indices = np.linspace(0, l-1, num=numberOfSamples, dtype=int)
    return array[indices]

if __name__ == "__main__":
    
    # Project Folders
    dataFolderPath = Path(r"C:\\Users\\vince\\OneDrive\\Documenti\\Università\\Magistrale\\Second Year\\Topic Highlights\\Final Project\\Materials\\EgoMotionSubtraction\\EgoMotionOFCode\\data\\test_unreal1\\image")
    outputFolderPath = dataFolderPath / "output"
    if not os.path.exists(outputFolderPath): os.makedirs(outputFolderPath)

    # Loading video frames
    log.setActive("LOADER")
    log.log("Loading video frames...")
    frames, width, height = loadImagesAsFrames(dataFolderPath)
    fps = 1 # FPS (int)

    framesQuantity = len(frames)
    if not frames: exit()

    # Setting up data
    log.log("Loading simulation frames...")
    focalLength = 256 # pixels (float)
    videoDepths = np.zeros((framesQuantity, height, width)) # meters (float)
    speedData = np.zeros((framesQuantity, 6)) # meters/sec (float)
    linearCameraSpeeds = speedData[:, 0:3] # meters/sec (float)
    angularCameraSpeeds = speedData[:, 3:6] # radians/sec (float)
    yawAngles = np.deg2rad([84, 87, 90, 93, 96, 99, 102]) # radians (float)
    yawAngles = np.unwrap(yawAngles, axis=0)
    consecutiveDifferences = np.diff(yawAngles, axis=0, prepend=yawAngles[0])
    angularCameraSpeeds[:, 1] = consecutiveDifferences*fps # radians/sec (float) (alias consecutiveDifferences/(1/fps))
    angularCameraSpeeds[0, 1] = angularCameraSpeeds[1, 1] # radians/sec (float) (alias consecutiveDifferences[0]/(1/fps))
    log.log(f"Yaw angles: {yawAngles}")
    log.log(f"Angular camera speeds: {angularCameraSpeeds[:, 1]}")

    plot(linearCameraSpeeds[:, 0], "Linear Camera Speed X")
    plot(linearCameraSpeeds[:, 1], "Linear Camera Speed Y")
    plot(linearCameraSpeeds[:, 2], "Linear Camera Speed Z")
    plot(angularCameraSpeeds[:, 0], "Angular Camera Speed X")
    plot(angularCameraSpeeds[:, 1], "Angular Camera Speed Y")
    plot(angularCameraSpeeds[:, 2], "Angular Camera Speed Z")

    # Computing optical flows
    log.setActive("OPTFLW")
    log.log("Computing the optical flows...")    
    naturalFlowFrames, egoFlowFrames, compensatedFlowFrames = computeOpticalFlows(
        frames, focalLength, videoDepths, linearCameraSpeeds, angularCameraSpeeds
    )

    # Saving optical flows to video
    log.setActive("SAVING")
    log.log(f"Now saving optical flows videos: {fps} FPS with {width}x{height} resolution")
    log.log("Saving the natural optical flow to video...")
    saveFramesToVideo(naturalFlowFrames, outputFolderPath / "naturalFlow.avi", fps, width, height)
    log.log("Saving the ego optical flow to video...")
    saveFramesToVideo(egoFlowFrames, outputFolderPath / "egoFlow.avi", fps, width, height)
    log.log("Saving the compensated optical flow to video...")
    saveFramesToVideo(compensatedFlowFrames, outputFolderPath / "compensatedFlow.avi", fps, width, height)

    cv2.destroyAllWindows() # Close all OpenCV windows
