
import numpy as np
import logger as log
from typing import List
import cv2

def loadVideoFrames(videoPath):

    if not videoPath: log.error("Error: the provided input video path is.. None!")
    # Launching the video capture system and getting related video parameters
    videoCapture = cv2.VideoCapture(videoPath)
    if not videoCapture.isOpened():
        log.error(f"Error: unable to open video at path \"{videoPath}\".")
    fps = videoCapture.get(cv2.CAP_PROP_FPS) # frame/sec (float)
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) # pixels (int)
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # pixels (int)
    # Reading all video frames
    processingVideo = True
    frames = []
    while processingVideo:
        successfullReading, frame = videoCapture.read()
        frames.append(frame)
        if not successfullReading:
            processingVideo = False
            break
    videoCapture.release()
    return frames, fps, width, height

def saveFramesToVideo(frames, outputVideoPath, fps, width, height):

    # Launching the video writer system
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # "fourcc" stands for "Four Character Code"
                                             # (it means that the code used to indicate the format of
                                             # the video, such as MJPG, must be of 4 characters)
                                             # Use MJPG for .avi files, XVID for .avi files, and MP4V for .mp4 files
    videoWriter = cv2.VideoWriter(outputVideoPath, fourcc, fps, (width, height))
    for frame in frames: videoWriter.write(frame)
    videoWriter.release()

def computeEgoMotionMatrices(width, height, focalLength):

    # Generating the xi and eta coordinates matrices
    xi = np.arange(-int(width/2), int(width/2) + 1) # pixels (int) (x)
    eta = np.arange(-int(height/2), int(height/2) + 1) # pixels (int) (y)
    xi = np.delete(xi, int(width/2))
    eta = np.delete(eta, int(height/2))
    # Generating the G and H matrices for the whole frame
    xiGrid, etaGrid = np.meshgrid(xi, eta)
    G = np.zeros((height, width, 2, 3))
    H = np.zeros((height, width, 2, 3))
    G[..., 0, 0] = focalLength
    G[..., 0, 2] = -xiGrid
    G[..., 1, 1] = focalLength
    G[..., 1, 2] = -etaGrid
    H[..., 0, 0] = (xiGrid*etaGrid)/focalLength
    H[..., 0, 1] = -focalLength - (xiGrid**2)/focalLength
    H[..., 0, 2] = etaGrid
    H[..., 1, 0] = focalLength + (etaGrid**2)/focalLength
    H[..., 1, 1] = -(xiGrid*etaGrid)/focalLength
    H[..., 1, 2] = -xiGrid
    return G,H

def visualizeFlow(frame, flow, decimation = 15, scale = 10, color = [255, 100, 30], thickness = 1):

    # Notice: Decimation controls the resolution of the optical flow visualization, allowing for
    #         a control in the number of drawn vectors to improve the readability of the image.
    # Notice: Scale controls the length of the optical flow vectors, allowing for a control in the
    #         magnitude of the drawn vectors to improve the readability of the image.
    frameOut = np.copy(frame)
    y = list(range(int(frameOut.shape[0])))[0::decimation]
    x = list(range(int(frameOut.shape[1])))[0::decimation]
    xv, yv = np.meshgrid(x, y)
    u = scale * flow[yv, xv, 0]
    v = scale * flow[yv, xv, 1]
    startPoints = np.array([xv.flatten(), yv.flatten()]).T.astype(int).tolist()
    endPoints = np.array([xv.flatten() - u.flatten(), yv.flatten() - v.flatten()]).T.astype(int).tolist()
    for i in range(len(startPoints)):
        cv2.arrowedLine(frameOut, tuple(startPoints[i]), tuple(endPoints[i]), color, thickness)
    return frameOut

def computeOpticalFlows(
        coloredFrames: List[np.ndarray],
        focalLength: float,
        videoDepths: List[np.ndarray],
        linearCameraSpeeds: List[List[float]],
        angularCameraSpeeds: List[List[float]]):
    
    framesWidth = coloredFrames[0].shape[1]
    framesHeight = coloredFrames[0].shape[0]

    # Unpacking video depths data and checking for consistency
    videoDepthsFramesQuantity = len(videoDepths)
    videoDepthsWidth = len(videoDepths[0])
    videoDepthsHeight = len(videoDepths[0][0])
    if videoDepthsFramesQuantity != len(coloredFrames) or videoDepthsWidth != framesWidth or videoDepthsHeight != framesHeight:
        log.error("Error: the provided video depths data is inconsistent with the video frames!")
    
    # Checking for consistency for the linear and angular camera speeds
    linearCameraSpeedsFramesQuantity = len(linearCameraSpeeds)
    linearCameraSpeedsDim = len(linearCameraSpeeds[0])
    angularCameraSpeedsFramesQuantity = len(angularCameraSpeeds)
    angularCameraSpeedsDim = len(angularCameraSpeeds[0])
    if linearCameraSpeedsFramesQuantity != len(coloredFrames) or linearCameraSpeedsDim != 3:
        log.error("Error: the provided linear camera speeds data is inconsistent with the video frames!")
    if angularCameraSpeedsFramesQuantity != len(coloredFrames) or angularCameraSpeedsDim != 3:
        log.error("Error: the provided angular camera speeds data is inconsistent with the video frames!")
    
    # Computing the optical flows for all frames
    naturalFlowFrames = []
    egoFlowFrames = []
    compensatedFlowFrames = []
    if coloredFrames is None: sys.exit()
    grayFrames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in coloredFrames]
    G,H = computeEgoMotionMatrices(framesWidth, framesHeight, focalLength)

    for j in range(1, len(grayFrames)):
        prevFrame = grayFrames[j - 1]
        currentFrame = grayFrames[j]
        flow = cv2.calcOpticalFlowFarneback(prev = prevFrame,    # Previous frame (grayscale image)
                                            next = currentFrame, # Current frame (grayscale image)
                                            flow = None,         # Output flow image; set to None to allow OpenCV to create it
                                            pyr_scale = 0.5,     # Image pyramid scale; a value between 0 and 1, controlling the resolution of each pyramid level
                                            levels = 3,          # Number of pyramid levels; higher values detect motion at larger scales
                                            winsize = 15,        # Window size for local averaging (in pixels)
                                            iterations = 3,      # Number of iterations for refining the flow estimation at each pyramid level
                                            poly_n = 5,          # Size of the pixel neighborhood used for polynomial expansion; higher values allow more complex motion
                                            poly_sigma = 1.2,    # Standard deviation of the Gaussian used for polynomial expansion; higher values make the flow smoother
                                            flags = 0)           # Flags to modify the algorithm behavior (usually 0 for default behavior)
        egoMotionU = np.dot(G, linearCameraSpeeds[j])
        egoMotionV = np.dot(H, angularCameraSpeeds[j])
        egoFlow = np.stack([egoMotionU, egoMotionV], axis=-1)
        compensatedFlow = flow - egoFlow
        naturalFlowFrames.append(visualizeFlow(coloredFrames[j], flow))
        egoFlowFrames.append(visualizeFlow(coloredFrames[j], egoFlow))
        compensatedFlowFrames.append(visualizeFlow(coloredFrames[j], compensatedFlow))

    return naturalFlowFrames, egoFlowFrames, compensatedFlowFrames