# Simple proxy metric: edge density (more structure after enhancement)
# (For demo; real mAP requires labels)

import cv2
import numpy as np

def edge_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.mean(edges)  # higher => more visible structure