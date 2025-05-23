# TIGER  <br>
Touchless Interactive Gesture Event Recognition  <br>

*a gesture tracking tool for interacting with your device remotely*  <br><br>
\\
---

## Overview <br>

This Python application uses MediaPipe to detect and track hand landmarks from a webcam feed. A Kivy-based GUI allows you to start and stop video capture and displays live hand-landmark overlays in real time. The last stable version used tkinter—see `_tk.py`.  <br>
\\
---

## Features <br>

- **Hand Detection & Tracking**  <br>
  Uses MediaPipe’s Hands solution to detect landmarks on one or multiple hands.<br>

- **Live Video Feed**  <br>
  Captures frames from your chosen webcam and overlays detected landmarks.<br><br>
  
---

## Installation & Usage<br>

1. Clone or download this repository.  <br>
2. Create & activate a venv:  <br>
   ```bash
   python -m venv venvname
   # Windows
   venvname\Scripts\activate
   # macOS/Linux
   source venvname/bin/activate

python gestures.py \  <br>

    --camera-index 1 \
    --detection-confidence 0.7 \  
    --tracking-confidence 0.7  
    
--camera-index: Which camera to use (default = 0).  <br>

--detection-confidence: Minimum confidence threshold for hand detection (default = 0.5). <br>  

--tracking-confidence: Minimum confidence threshold for landmark tracking (default = 0.5).  <br><br>

---

## How It Works  <br>
  
Initialization: A GestureTracker object opens the specified webcam and initializes the MediaPipe Hands module.  <br>

Processing: Each frame is converted from BGR to RGB, passed to MediaPipe for landmark detection, and updated in the GUI. <br><br>

\\
---  

  
## Contributing <br> 

Feel free to fork this project, add suggestions under issues, and open pull requests with improvements or additional gesture classification logic. <br><br>


## License  <br>

Distributed under the MIT License. See LICENSE file for more information.#     
  





g e s t u r e 
 
 
