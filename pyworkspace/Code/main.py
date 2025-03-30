
import cv2, os
import logger as log
import numpy as np
from miscellaneous import plot
from opticalFlow import loadVideoFrames, computeOpticalFlows, saveFramesToVideo
from objectDetection import performDetection, drawBoundingBoxes
from matplotlib import pyplot as plt

def resample(array, numberOfSamples):
    l = array.shape[0]
    if l == numberOfSamples: return array
    indices = np.linspace(0, l-1, num=numberOfSamples, dtype=int)
    return array[indices]

def opticalFlowTask(dataFolderPath, inputVideoPath, outputFolderPath):

    log.setActive("OPTFLW")
    log.log("Now running the optical flow task...")

    # Loading video frames
    log.setActive("LOADER")
    log.log("Loading video frames...")
    frames, fps, width, height = loadVideoFrames(inputVideoPath)
    framesQuantity = len(frames)
    if not frames: exit()
    log.log(f"Loaded {framesQuantity} frames with {fps} FPS and {width}x{height} resolution")

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
    log.log(f"Now saving optical flows videos: {fps} FPS with {width}x{height} resolution")
    log.log("Saving the natural optical flow to video...")
    saveFramesToVideo(naturalFlowFrames, outputFolderPath + os.sep + "naturalFlow.avi", fps)
    log.log("Saving the ego optical flow to video...")
    saveFramesToVideo(egoFlowFrames, outputFolderPath + os.sep + "egoFlow.avi", fps)
    log.log("Saving the compensated optical flow to video...")
    saveFramesToVideo(compensatedFlowFrames, outputFolderPath + os.sep + "compensatedFlow.avi", fps)

    cv2.destroyAllWindows() # Close all OpenCV windows

def objectDetectionTask(dataFolderPath, inputVideoPath, outputFolderPath):
    
    log.setActive("ODYOLO")
    log.log("Now running the object detection task...")

    # Loading video frames
    log.setActive("LOADER")
    log.log("Loading video frames...")
    frames, fps, width, height = loadVideoFrames(inputVideoPath)
    framesQuantity = len(frames)
    if not frames: exit()
    log.log(f"Loaded {framesQuantity} frames with {fps} FPS and {width}x{height} resolution")

    # Setting up data
    log.log("Loading simulation data...")
    videoDepths = resample(np.load(dataFolderPath + os.sep + "depth_tensor.npy"), framesQuantity) # meters (float)

    # Unpacking video depths data and checking for consistency
    videoDepthsFramesQuantity = len(videoDepths)
    videoDepthsWidth = len(videoDepths[0][0])
    videoDepthsHeight = len(videoDepths[0])
    if videoDepthsFramesQuantity != len(frames) or videoDepthsWidth != width or videoDepthsHeight != height:
        log.error("Error: the provided video depths data is inconsistent with the video frames!")

    # Cutting data...
    startingTime = 14
    endingTime = 29
    startFrame = int(startingTime*fps)
    endFrame = int(endingTime*fps) - 6
    frames = frames[startFrame:endFrame]
    videoDepths = videoDepths[startFrame:endFrame]
    log.log(f"Cutted to {len(frames)} frames, from {startFrame} to {endFrame} of the original frames")

    # Performing YOLOv8 object detection
    log.setActive("ODYOLO")
    log.log("Performing YOLOv8 object detection...")
    ybboxes = performDetection(frames, dataFolderPath, yoloModelFileName = "yolov8n-seg.pt")

    # Drawing bounding boxes on video frames
    log.setActive("DRWBOX")
    log.log("Drawing bounding boxes on video frames...")
    framesWithBoxes = drawBoundingBoxes(frames, ybboxes, videoDepths)

    # Saving optical flows to video
    log.setActive("SAVING")
    log.log(f"Now saving result to video ({fps} FPS with {width}x{height} resolution)...")
    saveFramesToVideo(framesWithBoxes, outputFolderPath + os.sep + "output.avi", fps)

    # Plotting the results
    log.setActive("PLTING")
    log.log("Plotting depth result on graph...")
    depthsBoxCenter = []
    depthsBlobCenter = []
    depthsProjectedBoxCenterOnBlob = []
    depthsBlobMedians = []
    depthsImprovedBlobMedians = []
    for idx, ybbox in enumerate(ybboxes):
        if ybbox is None:
            depthsBoxCenter.append(0)
            depthsBlobCenter.append(0)
            depthsProjectedBoxCenterOnBlob.append(0)
            depthsBlobMedians.append(0)
            depthsImprovedBlobMedians.append(0)
        else:
            xCenter = ybbox.xCenter
            yCenter = ybbox.yCenter
            blobCenterX = ybbox.blobCenterX
            blobCenterY = ybbox.blobCenterY

            depthsBoxCenter.append(videoDepths[idx][yCenter, xCenter])
            depthsBlobCenter.append(videoDepths[idx][blobCenterY, blobCenterX])
            depthsProjectedBoxCenterOnBlob.append(videoDepths[idx][ybbox.yCenterProjected, ybbox.xCenterProjected])

            depthsImprovedBlobMedians.append(ybbox.generateDepth(videoDepths[idx]))

            """
            boxDepths = videoDepths[idx][ybbox.y:ybbox.y+ybbox.h, ybbox.x:ybbox.x+ybbox.w]
            boxDepths = boxDepths[ybbox.blobGrid == 1]
            boxDepths = boxDepths[np.isfinite(boxDepths)]
            if len(boxDepths) > 0:
                depth = np.median(boxDepths)
                print(depth)
                depthsBlobMeans.append(depth)
            
            boxDepths = videoDepths[idx][ybbox.y:ybbox.y+ybbox.h, ybbox.x:ybbox.x+ybbox.w]
            boxDepths = boxDepths[ybbox.getOtsuImprovedBlob(videoDepths[idx]) == 1]
            boxDepths = boxDepths[np.isfinite(boxDepths)]
            if len(boxDepths) > 0:
                depth = np.median(boxDepths)
                depthsImprovedBlobMeans.append(depth)"
            """
    
    plt.figure(figsize=(10, 5))
    plt.plot(depthsBoxCenter, linestyle='-', color='green', label='Accordingly to Box Center')
    plt.plot(depthsProjectedBoxCenterOnBlob, linestyle='-', color='purple', label='Accordingly to Projected Box Center')
    plt.plot(depthsBlobCenter, linestyle='-', color='orange', label='Accordingly to Blob Center')
    # plt.plot(depthsBlobMedians, linestyle='-', color='red', label='From Blob Values Median')
    plt.plot(depthsImprovedBlobMedians, linestyle='-', color='blue', label='From Blob Depth Values Median')
    plt.title("Depth")
    plt.xlabel(f"frames ({fps} FPS)")
    plt.ylabel("depth (m)")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # Project Folders
    dataFolderPath = "C:\\Users\\vince\\OneDrive\\Documenti\\Università\\Magistrale\\Second Year\\Topic Highlights\\Final Project\\pyworkspace\\Code\\Data\\objectDetection"
    inputVideoPath = dataFolderPath + os.sep + "input.avi"
    outputFolderPath = dataFolderPath + os.sep + ""
    if not os.path.exists(outputFolderPath): os.makedirs(outputFolderPath)

    objectDetectionTask(dataFolderPath, inputVideoPath, outputFolderPath)

    log.log("All completed with success!")
