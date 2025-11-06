#!/bin/bash
echo "Starting Attendance System..."
python3 attendance_system.py
```
(Make it executable: `chmod +x run.sh`)

### 5. **requirements.txt**
```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
pandas>=1.2.0
Pillow>=8.0.0
numpy>=1.19.0
```

### 6. **README.txt**
```
ATTENDANCE SYSTEM - INSTALLATION & USAGE GUIDE
==============================================

PREREQUISITES:
1. Python 3.6 or higher must be installed
   Download from: https://www.python.org/downloads/

INSTALLATION:
1. Unzip this folder
2. Open Command Prompt/Terminal in this folder
3. Run: pip install -r requirements.txt
4. Wait for all libraries to install

RUNNING THE SYSTEM:
- Windows: Double-click "run.bat"
- Mac/Linux: Double-click "run.sh" (or run in terminal)

FIRST TIME SETUP:
1. Click "Take Images" button
2. Enter student ID and Name
3. Press the button and look at the camera (50 images will be captured)
4. Press 'Q' to finish early if needed
5. Click "Save Profile" to train the system

TAKING ATTENDANCE:
1. Click "Take Attendance"
2. Students look at the camera
3. System automatically marks attendance
4. Press 'Q' when done
5. Check the Attendance folder for CSV files

TROUBLESHOOTING:
- If camera doesn't work, make sure no other app is using it
- If "cv2.face not found" error appears, reinstall opencv-contrib-python
- Files are created in StudentDetails/ and Attendance/ folders

For support, contact: [Your Email/Contact]