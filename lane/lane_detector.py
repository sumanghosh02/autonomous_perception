import cv2
import numpy as np

def detect_lanes(frame):
    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Region of Interest (ONLY ROAD AREA)
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height*0.6)),
        (0, int(height*0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough Transform (STRONG FILTER)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 150,
                            minLineLength=150, maxLineGap=50)

    # Draw only few strong lines
    if lines is not None:
        for i, line in enumerate(lines[:4]):  # LIMIT to 4 lines
            x1, y1, x2, y2 = line[0]

            # Only near-horizontal lines (road lanes)
            if abs(y2 - y1) < 50:
                cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)

    return frame