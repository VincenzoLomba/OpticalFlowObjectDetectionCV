
import cv2, os
import logger as log
from opticalFlow import loadVideoFrames, computeOpticalFlows, saveFramesToVideo

if __name__ == "__main__":
    
    # Setting up data
    inputVideoPath = ""
    outputFolderPath = ""
    focalLength = 522.196 # pixels (float)

    # Loading video frames
    log.setActive("LOADER")
    log.log("Loading video frames...")
    frames, fps, width, height = loadVideoFrames(inputVideoPath)
    if not frames: exit()

    # Computing optical flows
    log.setActive("OPTFLW")
    log.log("Computing the optical flows...")    
    naturalFlowFrames, egoFlowFrames, compensatedFlowFrames = computeOpticalFlows(
        frames, width, height, focalLength, videoDepths, linearCameraSpeeds, angularCameraSpeeds
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
