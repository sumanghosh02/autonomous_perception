import cv2
from config import VIDEO_PATH, USE_NOISE_ATTACK, CONFIDENCE

from detection.detector import Detector
from lane.lane_detector import detect_lanes
from enhancement.enhancer import enhance
from security.attack import add_noise
from utils.fps import FPS
from evaluation.metrics import edge_density

# Initialize
detector = Detector(conf=CONFIDENCE)
fps_counter = FPS()

# Video
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- ORIGINAL ----------------
    original = frame.copy()
    orig_detect = detector.detect(original.copy())

    # ---------------- ROBUST PIPELINE ----------------
    # Step 1: Enhancement
    enhanced = enhance(frame)

    # Step 2: Add noise (simulate disturbance)
    if USE_NOISE_ATTACK:
        attacked = add_noise(enhanced)
    else:
        attacked = enhanced

    # Step 3: Detection
    robust_detect = detector.detect(attacked)

    # Step 4: Adaptive lane detection (OPTION 2)
    if USE_NOISE_ATTACK:
        # Skip lane detection in bad conditions
        robust_final = robust_detect
    else:
        # Apply lane detection only in good conditions
        robust_final = detect_lanes(robust_detect)

    # ---------------- METRICS ----------------
    fps = fps_counter.tick()
    edge_orig = edge_density(original)
    edge_enh = edge_density(enhanced)

    # ---------------- TEXT ----------------
    cv2.putText(orig_detect, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(robust_final, "Robust Pipeline", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(robust_final, f"FPS: {fps}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(robust_final, f"Edge(orig): {edge_orig:.1f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(robust_final, f"Edge(enh): {edge_enh:.1f}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ---------------- DISPLAY ----------------
    top = cv2.resize(orig_detect, (640, 360))
    bottom = cv2.resize(robust_final, (640, 360))

    display = cv2.vconcat([top, bottom])

    cv2.imshow("Autonomous Perception (Top: Original | Bottom: Robust)", display)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()