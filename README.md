### Title
### Vacant Parking Slot Detection using Python and OpenCV

### Introduction
This project implements a computer vision pipeline to automatically detect vacant parking slots from camera footage using Python and OpenCV. The system processes video frames, identifies predefined parking regions, and determines whether each slot is occupied or free based on visual evidence.

### Task performed
Read frames from a live camera or recorded video and apply preprocessing steps such as resizing, grayscale conversion, and blurring.

For each predefined parking slot region, analyze pixel statistics or foreground activity to classify it as occupied or vacant, and overlay visual indicators on the frame.

### Algorithms
Background modeling or frame differencing to highlight changes in parking regions compared with their empty reference state.

Thresholding and contour or pixel-density analysis to estimate whether a car is present in each slot, followed by simple rules to label slots as free or occupied.

### Implementation details
Technology stack

Python 3.x

OpenCV 

NumPy

### Prerequisites

Python 3.x 

Git

### 1. Clone the repository

```bash
git clone https://github.com/Sreeja-01/Vacant_parking_slot_detection-python-opencv.git
cd Vacant_parking_slot_detection-python-opencv
```
### 2. Create and activate a virtual environment (optional but recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
Linux/macOS:

bash
python -m venv venv
source venv/bin/activate
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
Otherwise, install the core libraries manually:

bash
pip install opencv-python
pip install numpy
```
### 4. Configure parking slots and video source

Set the input video path or camera index in the main script (for example, parking_detection.py).

Define the coordinates of each parking slot polygon/rectangle in a configuration section or file so the algorithm knows which regions to evaluate.

### 5. Run the application

```bash
python parking_detector.py
```

Results1

https://github.com/user-attachments/assets/50218b51-c030-4d90-a79b-b526438a45cd

Produces annotated video frames where each parking slot is highlighted and labeled as VACANT or OCCUPIED, enabling quick visual assessment of available spaces.

Demonstrates how classical OpenCV-based image processing can be used to build a practical smart-parking helper that can be extended to real-time deployment with minimal additional work.
