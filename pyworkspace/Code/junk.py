

from opticalFlow import loadVideoFrames, saveFramesToVideo
import os

# Loading video frames
inputVideoPath = "C:\\Users\\vince\\Downloads"
frames, fps, width, height = loadVideoFrames(inputVideoPath + os.sep + "depth_output 2.avi")
framesQuantity = len(frames)
# Cutting video
startingTime = 14
endingTime = 29
startingFrame = int(startingTime*fps)
endingFrame = int(endingTime*fps)-6
if startingFrame < 0: startingFrame = 0
if endingFrame > framesQuantity: endingFrame = framesQuantity
frames = frames[startingFrame:endingFrame]
saveFramesToVideo(frames, inputVideoPath + os.sep + "depths.avi", fps)