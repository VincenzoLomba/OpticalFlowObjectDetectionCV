
from typing import List
from ultralytics import YOLO
import os, cv2
import logger as log
import numpy as np

class YBBox:

    def __init__(self, xCenter, yCenter, w, h, confidence, blobGrid):

        self.x = int(xCenter - w / 2)
        self.y = int(yCenter - h / 2)
        self.xCenter = int(xCenter)
        self.yCenter = int(yCenter)
        self.w = int(w)
        self.h = int(h)
        self.confidence = confidence
        self.blobGrid = blobGrid
        if self.blobGrid is not None and np.count_nonzero(self.blobGrid) > 0:
            coords = np.column_stack(np.where(self.blobGrid == 1))
            self.blobCenterX = int(coords[:, 1].mean()) + self.x
            self.blobCenterY = int(coords[:, 0].mean()) + self.y
        else:
            self.blobCenterX, self.blobCenterY = self.xCenter, self.yCenter
        self.xCenterProjected, self.yCenterProjected = self.projectCenterOnBlob()
        self.depth = None

    def projectCenterOnBlob(self):

        if self.blobGrid is None or np.count_nonzero(self.blobGrid) == 0: return self.xCenter, self.yCenter
        erosionKernel = np.ones((3, 3), np.uint8) # Performs erosion to ensure project it-on the blob perimeter
        erodedBlob = self.blobGrid.astype(np.uint8)
        erodedBlob = cv2.erode(erodedBlob, erosionKernel, iterations = 10)  
        coords = np.column_stack(np.where(erodedBlob == 1))
        distances = np.sqrt((coords[:, 0] - (self.yCenter - self.y))**2 + (coords[:, 1] - (self.xCenter - self.x))**2)
        if len(distances) == 0:
            coords = np.column_stack(np.where(self.blobGrid == 1))
            distances = np.sqrt((coords[:, 0] - (self.yCenter - self.y))**2 + (coords[:, 1] - (self.xCenter - self.x))**2)
        nearestIdx = np.argmin(distances)
        nearestY, nearestX = coords[nearestIdx]
        return self.x + nearestX, self.y + nearestY
    
    def getOtsuImprovedBlob(self, fullFrameDepths):

        if self.blobGrid is None or np.count_nonzero(self.blobGrid) == 0: return None
        boxDepths = fullFrameDepths[self.y:self.y+self.h, self.x:self.x+self.w]
        # Get the depths of the bounding box pixels that also belong to the blob
        objectDepths = boxDepths[self.blobGrid == 1]
        # Normalize the depths to the range [0, 255] for Otsu's thresholding
        clippedDepths = np.clip(objectDepths, np.percentile(objectDepths, 5), np.percentile(objectDepths, 95))
        normalizedForOtsuDepths = cv2.normalize(clippedDepths, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply Otsu's thresholding to find the optimal threshold value
        otsuThresholdValue, _ = cv2.threshold(normalizedForOtsuDepths, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Calculate the threshold value in the original depth range
        thresholdValue = (otsuThresholdValue / 255) * (objectDepths.max() - objectDepths.min()) + objectDepths.min()
        # Create a mask for the blob pixels with depth values below the threshold
        filteredMask = np.zeros_like(self.blobGrid, dtype=np.uint8)
        validPixels = np.logical_and((self.blobGrid == 1), (boxDepths <= thresholdValue))
        filteredMask[validPixels] = 1
        if np.count_nonzero(filteredMask) == 0: filteredMask = self.blobGrid
        boxDepths = boxDepths[filteredMask == 1]
        boxDepths = boxDepths[np.isfinite(boxDepths)]
        self.depth = np.median(boxDepths)
        return filteredMask
    
    def generateDepth(self, fullFrameDepths):
        if self.depth is None: self.getOtsuImprovedBlob(fullFrameDepths)
        return self.depth

def performDetection(
        frames: List[np.ndarray],         # List of frames (numpy arrays) to process
        dataFolderPath,                   # Path to the folder containing the YOLOv8 model file
        yoloModelFileName = "yolov8n.pt", # YOLOv8 model file name ("n" stands for "nano", the lightest version)
        confidenceThreshold = 0.75        # Confidence threshold for detection
                                          # (BoundBoxes are shown only if thier confidence is larger w.r.t that threshold)
    ):

    log.log("Loading YOLOv8 model...")

    # Environment variables configuration (for OpenMP and Intel MKL optimization)
    # relevant for optimizing YOLOv8's parallel processing on multi-core CPUs
    os.environ['OMP_NUM_THREADS'] = '14'                        # Limits OpenMP to 14 threads
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0' # Thread-core binding strategy (Intel CPUs):
                                                                # granularity=fine : Threads pinned to specific CPU cores
                                                                # compact          : Threads packed together (better cache locality)
                                                                # 1                : enables verbose affinity warnings (0=disabled)
                                                                # 0                : assigns threads consecutively from core 0 and then upwards
                                                                #                    (if 1, assigns threads to cores in a round-robin fashion)

    # Check for YOLOv8 model file presence
    yoloModelPath = dataFolderPath + os.sep + yoloModelFileName
    if not os.path.exists(yoloModelPath):
        log.error(f"File {yoloModelFileName} missing from folder {dataFolderPath}!")

    # Actually loading YOLOv8 model
    model = YOLO(dataFolderPath + os.sep + yoloModelFileName, verbose = False)
    model.fuse() # Layer fusion (optional, but recommended for performance), implies faster inference time) 

    log.log("Performing YOLOv8 detection on all frames...")
    ybboxes = []
    progressLabel = -1
    for j in range(0, len(frames)):

        frame = frames[j]
        yoloResults = model.predict(
            source  = frame,               # Input frame for detection (numpy array/HWC format)
            conf    = confidenceThreshold, # Minimum confidence score (0.65) to accept detection
            classes = [0],                 # Only detect people (class 0 in COCO dataset)
            verbose = False                # Disable console output (cleaner execution)
        )

        # YOLO results elaboration...
        ybbox = None
        for singleResult in yoloResults:
            boxes = singleResult.boxes.xywh.cpu().numpy()
            confidences = singleResult.boxes.conf.cpu().numpy()
            masks = singleResult.masks.data.cpu().numpy() if singleResult.masks else None
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                xCenter, yCenter, w, h = box
                if not ybbox:
                    blobGrid = None
                    if masks is not None and idx < len(masks):
                        mask = masks[idx]
                        x1 = max(0, int(xCenter - w / 2))
                        y1 = max(0, int(yCenter - h / 2))
                        x2 = min(mask.shape[1], int(xCenter + w / 2))
                        y2 = min(mask.shape[0], int(yCenter + h / 2))
                        blobGrid = mask[y1:y2, x1:x2]
                        if blobGrid.shape != (int(h), int(w)):
                            blobGrid = cv2.resize(blobGrid, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
                        blobGrid = np.where(blobGrid > 0.4, 1, 0)
                    else:
                        log.log(f"Mask not found for frame number {j+1}!")
                    ybbox = YBBox(xCenter, yCenter, w, h, conf, blobGrid)
                else:
                    log.log(f"Multiple objects detections for frame number {j+1}!")
                    break
        ybboxes.append(ybbox)

        # Updating "progress bar"
        progress = (int)(j/len(frames)*100)
        if progress % 10 == 0 and progress != progressLabel:
            progressLabel = progress
            log.log("Progress: " + str(progress) + "%")

    return ybboxes

def drawBoundingBoxes(
        frames: List[np.ndarray],
        ybboxes: List[YBBox],
        videoDepths: List[np.ndarray]
    ):

    framesWithBoxes = []
    progressLabel = -1
    progressLabel = -1
    colorGreen = (0, 255, 0)
    purpleColor = (128, 0, 128)
    orangeColor = (0, 165, 255)

    for j in range(0, len(frames)):

        # Updating "progress bar"
        progress = (int)(j/len(frames)*100)
        if progress % 10 == 0 and progress != progressLabel:
            progressLabel = progress
            log.log("Progress: " + str(progress) + "%")
        
        # Retrieving current frame and related bounding box and depths
        frame = frames[j].copy()
        ybbox = ybboxes[j]
        if not ybbox:
            framesWithBoxes.append(frame)
            continue
        x = ybbox.x
        y = ybbox.y
        w = ybbox.w
        h = ybbox.h
        xCenter = ybbox.xCenter
        yCenter = ybbox.yCenter
        xCenterProjected = ybbox.xCenterProjected
        yCenterProjected = ybbox.yCenterProjected
        blobCenterX = ybbox.blobCenterX
        blobCenterY = ybbox.blobCenterY
        confidence = ybbox.confidence
        blobGrid = ybbox.blobGrid
        depths = videoDepths[j] if videoDepths is not None else None

        # Drawing the improved blob perimeter
        # if depths is not None:
        #     contours, _ = cv2.findContours(ybbox.getOtsuImprovedBlob(depths).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for contour in contours: cv2.drawContours(frame, [contour + [x, y]], -1, purpleColor, 2)

        # Drawing the blob perimeter
        contours, _ = cv2.findContours(blobGrid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours: cv2.drawContours(frame, [contour + [x, y]], -1, orangeColor, 2)

        # Drawing bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), colorGreen, 2)

        # Drawing bounding box center point projected on the blob
        cv2.circle(frame, (xCenterProjected, yCenterProjected), 4, purpleColor, -1)

        # Drawing the bounding box center point
        cv2.circle(
            frame,              # Input image/frame where circle will be drawn
            (xCenter, yCenter), # Center coordinates (calculated as x + w//2, y + h//2)
            4,                  # Circle radius in pixels (4px = small visible dot)
            colorGreen,         # Circle color (matches bounding box color)
            -1                  # Thickness (-1 = filled circle, positive = outline thickness)
        )

        # Drawing blob center point
        cv2.circle(frame, (blobCenterX, blobCenterY), 4, orangeColor, -1)
        
        # Drawing label on top of the bounding box
        label = f"Person: {confidence:.2f}%"
        depth = depths[yCenter, xCenter] if depths is not None else None
        if depth is not None: label = f"Person: {confidence:.2f}% {depth:.2f}m"
        cv2.putText(
            frame,                    # Input image/frame where text will be drawn
            label,                    # Text string to display (e.g., "Person: 0.95")
            (x, y - 5),               # Text position: 5px above top-left box corner (x,y)
            cv2.FONT_HERSHEY_SIMPLEX, # Font type (clean, sans-serif)
            0.5,                      # Font scale (0.5 = 50% of base size)
            colorGreen,               # Text color (matches bounding box)
            1                         # Thickness (1px)
        )

        framesWithBoxes.append(frame)

    return framesWithBoxes