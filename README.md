# 🚦 Smart Traffic Control System using YOLOv5

## 📌 About the Project
This project implements an **AI-powered Smart Traffic Light System** using **YOLOv5** for real-time vehicle detection.  
Unlike traditional traffic lights that work on **fixed timers**, this system is *adaptive* — it dynamically adjusts the green light duration based on the **number of vehicles waiting** in each lane.  

The project simulates how **intelligent traffic management** can reduce congestion and improve road efficiency.

---

## 📖 Description
- Input: A traffic video of an intersection.  
- Processing:  
  - Detect vehicles using YOLOv5.  
  - Count vehicles in **North-South (NS)** and **East-West (EW)** lanes.  
  - Adjust **green light duration dynamically** depending on vehicle density.  
  - Traffic signal changes follow a **state machine** (Green → Yellow → All-Red → Switch).  
- Output: A new video showing:  
  - Bounding boxes around vehicles.  
  - Regions of Interest (ROIs) for each lane.  
  - Traffic lights (with timers).  
  - Vehicle counts and dynamic green times.  

---

## ✨ Features
- ✅ Real-time **vehicle detection** using YOLOv5.  
- ✅ Adaptive green light duration (more cars = longer green).  
- ✅ **State machine logic** ensures realistic light switching.  
- ✅ Safety with **yellow and all-red phases**.  
- ✅ Output video with visual overlays (signals, timers, counts).  

---

## 📊 Flowchart
<img height="500" alt="START (9)" src="https://github.com/user-attachments/assets/9b0d2d9a-0361-4a14-baed-2a1c745b80b4" />

---

## ✨ Installation

1. Clone the YOLOv5 Repository
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   ```

2. Install required dependencies
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision opencv-python numpy roboflow
   ```
