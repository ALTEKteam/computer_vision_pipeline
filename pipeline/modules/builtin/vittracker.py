import cv2 as cv

class VitTracker:
    def __init__(self):
        # Initialization of the VIT Tracker
        try:
            net = "/home/furkan/Desktop/CS/altek/pipeline/models/vitTracker.onnx"  # VIT model yolu
            model = cv.dnn.readNet(net)
            self.tracker = cv.TrackerVit_create(model,tracking_score_threshold=0.5)
            print(">> VIT Tracker has been initialized successfully.")
        except AttributeError:
            print("ERROR: there is no vittrack implementation in the current OpenCV version. Please run 'pip install opencv-contrib-python' to install vittrack or update the version.")
            self.tracker = None

    def initialize(self, frame, bbox):
        """
        locks the tracker onto the initial bounding box in the first frame.
        Input:
            - frame: The initial video frame (numpy array).
            - bbox format: [x, y, w, h] (int or float can be accepted, but will be converted to int)
        output: None
        """
        if self.tracker is None: return
        
        # convertion to int if necessary
        bbox = tuple(map(int, bbox))
        
        # it can be necessary to re-create the tracker for each initialization,  not reset properly with init.
        if (not self.tracker):   self.tracker = cv.TrackerVit_create() 
        
        self.tracker.init(frame, bbox)

    def track(self, frame):
        """
        Predicts the next position of the target in next frame
        Input: frame (Numpy array)
        Output: [x, y, w, h] or None (if lost)
        """
        if self.tracker is None: return None

        success, bbox = self.tracker.update(frame)
        print("Success:", success, "BBox:", bbox)
        if success:
            # OpenCV returns tuple. convert it to a list of ints
            return {"target_bbox": bbox, "best_score": self.tracker.getTrackingScore()}
        else:
            return None