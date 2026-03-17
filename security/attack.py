import numpy as np

def add_noise(frame, sigma=10):
    noise = np.random.normal(0, sigma, frame.shape).astype(np.int16)
    attacked = frame.astype(np.int16) + noise
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked