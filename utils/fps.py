import time

class FPS:
    def __init__(self):
        self.prev = time.time()

    def tick(self):
        curr = time.time()
        fps = 1.0 / max(curr - self.prev, 1e-6)
        self.prev = curr
        return int(fps)