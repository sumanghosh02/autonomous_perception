# 🚗 Robust Autonomous Car Perception System

## 📌 Overview

This project presents a real-time autonomous vehicle perception system that not only detects objects but also improves detection under challenging conditions and evaluates system robustness under disturbances.

---

## 🎯 Features

* Object Detection (YOLOv8)
* Image Enhancement (low-light improvement)
* Noise Simulation (disturbance testing)
* Robustness Evaluation (fail → recover)
* Real-time FPS display
* Adaptive lane detection (disabled in noisy conditions)

---

## 🧠 Unique Contribution

This project focuses on **robustness in perception systems**, improving detection performance under adverse conditions and evaluating system reliability, which is often not addressed in traditional projects.

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/your-username/autonomous-perception.git
cd autonomous-perception
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Input Video

Place your video here:

```
data/sample_video.mp4
```

---

## ▶️ Run

```bash
python main.py
```

---

## 🧪 Demonstration

* Top: Normal detection
* Bottom: Robust pipeline
* Shows performance under disturbance

---

## ⚠️ Requirements

* Python 3.10 or 3.11
* Internet (for model download)

---

## 🛠️ Tech Stack

* Python
* OpenCV
* PyTorch
* YOLOv8

---

## ❗ Common Issues

* Torch not installing → Use Python 3.11
* Video not loading → Check path
* Slow FPS → Use lightweight model

---

## 🚀 Future Work

* Multi-sensor fusion
* Adversarial attack defense
* Object tracking

---

## 👤 Author

Suman Ghosh
GitHub: https://github.com/sumanghosh02
