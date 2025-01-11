import cv2

class ObjectTracker:
    def __init__(self):
        self.prev_keypoints = None

    def track(self, prev_frame, curr_frame, mask):
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        keypoints = cv2.goodFeaturesToTrack(gray_prev, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7)
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, keypoints, None)
        return new_points
