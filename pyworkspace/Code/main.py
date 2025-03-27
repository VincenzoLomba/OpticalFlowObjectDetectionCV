
import cv2, os
import logger as log
import numpy as np
from opticalFlow import loadVideoFrames, computeOpticalFlows, saveFramesToVideo

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
    if not frames: exit()

    # Setting up data
    focalLength = 522.196 # pixels (float)
    videoDepths = np.zeros((len(frames), height, width)) # meters (float)
    linearCameraSpeeds = np.zeros((len(frames), 3))      # meters/sec (float)
    angularCameraSpeeds = np.zeros((len(frames), 3))     # radians/sec (float)

    # Computing optical flows
    log.setActive("OPTFLW")
    log.log("Computing the optical flows...")    
    naturalFlowFrames, egoFlowFrames, compensatedFlowFrames = computeOpticalFlows(
        frames, focalLength, videoDepths, linearCameraSpeeds, angularCameraSpeeds
    )

    # Saving optical flows to video
    log.setActive("SAVING")
    log.log("Saving the natural optical flow to video...")
    saveFramesToVideo(naturalFlowFrames, outputFolderPath + os.sep + "naturalFlow.avi", fps, width, height)
    log.log("Saving the ego optical flow to video...")
    saveFramesToVideo(egoFlowFrames, outputFolderPath + os.sep + "egoFlow.avi", fps, width, height)
    log.log("Saving the compensated optical flow to video...")
    saveFramesToVideo(compensatedFlowFrames, outputFolderPath + os.sep + "compensatedFlow.avi", fps, width, height)

    cv2.destroyAllWindows() # Close all OpenCV windows
