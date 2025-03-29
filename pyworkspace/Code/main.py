
import cv2, os
import logger as log
import numpy as np
from miscellaneous import plot
from opticalFlow import loadVideoFrames, computeOpticalFlows, saveFramesToVideo, loadCameraVelocitiesFromStationaryEnvironment

def resample(array, numberOfSamples):
    l = array.shape[0]
    if l == numberOfSamples: return array
    indices = np.linspace(0, l-1, num=numberOfSamples, dtype=int)
    return array[indices]

if __name__ == "__main__":
    
    # Project Folders
    dataFolderPath = "C:\\Users\\vince\\OneDrive\\Documenti\\Università\\Magistrale\\Second Year\\Topic Highlights\\Final Project\\pyworkspace\\Code\\Data"
    inputVideoPath = dataFolderPath + os.sep + "input.avi"
    outputFolderPath = dataFolderPath + os.sep + "output"
    if not os.path.exists(outputFolderPath): os.makedirs(outputFolderPath)

    # Loading video frames
    log.setActive("LOADER")
    log.log("Loading video frames...")
    frames, fps, width, height = loadVideoFrames(inputVideoPath)
    framesQuantity = len(frames)
    if not frames: exit()

    # Setting up data
    log.log("Loading simulation data...")
    focalLength = 522.196 # pixels (float)
    videoDepths = resample(np.load(dataFolderPath + os.sep + "depth_tensor.npy"), framesQuantity) # meters (float)
    speedData = resample(np.load(dataFolderPath + os.sep + "camera_velocities.npy"), framesQuantity) # meters/sec (float)
    linearCameraSpeeds = speedData[:, 0:3] # meters/sec (float)
    angularCameraSpeeds = speedData[:, 3:6] # radians/sec (float)

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
    log.log("Saving the natural optical flow to video...")
    saveFramesToVideo(naturalFlowFrames, outputFolderPath + os.sep + "naturalFlow.avi", fps)
    log.log("Saving the ego optical flow to video...")
    saveFramesToVideo(egoFlowFrames, outputFolderPath + os.sep + "egoFlow.avi", fps)
    log.log("Saving the compensated optical flow to video...")
    saveFramesToVideo(compensatedFlowFrames, outputFolderPath + os.sep + "compensatedFlow.avi", fps)

    cv2.destroyAllWindows() # Close all OpenCV windows
