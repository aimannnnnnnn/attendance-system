import tkinter as tk
from tkinter import ttk, messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os, csv, numpy as np
from PIL import Image
import pandas as pd
import datetime, time
from collections import defaultdict

# --------------------------- UTILS ---------------------------

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror('Error', 'Missing "haarcascade_frontalface_default.xml" file!')
        window.destroy()

# ---------------------- FACE FUNCTIONS -----------------------

def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    Id = txt.get().strip()
    name = txt2.get().strip()

    if not Id or not name:
        mess.showwarning('Missing info', 'Enter both ID and Name!')
        return

    if not name.replace(" ", "").isalpha():
        mess.showerror('Invalid Name', 'Name should only contain alphabets!')
        return

    serial = 1
    exists = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists:
        with open("StudentDetails/StudentDetails.csv", 'r') as f:
            serial = sum(1 for _ in f) // 2

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    sampleNum = 0
    skip_frames = 3  # Process every 3rd frame for better performance
    frame_count = 0

    mess.showinfo('Instructions', "Press 'Q' to stop capturing early.")
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
        frame_count += 1

        # Only process every nth frame for face detection
        if frame_count % skip_frames == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Optimized detection parameters
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.imwrite(
                    f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg",
                    gray[y:y + h, x:x + w]
                )
                cv2.putText(img, f"Captured: {sampleNum}/50", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Taking Images (Press Q to quit)", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or sampleNum >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['SERIAL NO.', 'ID', 'NAME'])
        writer.writerow([serial, Id, name])

    message1.config(text=f"Images captured for ID: {Id}")

# ------------------------- TRAINING --------------------------

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        mess.showerror('Error', 'cv2.face not found! Install opencv-contrib-python.')
        return

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    try:
        imagePaths = [os.path.join("TrainingImage", f) for f in os.listdir("TrainingImage") 
                     if f.endswith('.jpg') or f.endswith('.png')]
        
        if not imagePaths:
            mess.showerror('Error', 'No training images found!')
            return

        faces, IDs = [], []
        for imagePath in imagePaths:
            try:
                gray_img = Image.open(imagePath).convert('L')
                np_img = np.array(gray_img, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(np_img)
                IDs.append(Id)
            except Exception as e:
                print(f"Error processing {imagePath}: {e}")
                continue

        if not faces:
            mess.showerror('Error', 'No valid training images found!')
            return

        recognizer.train(faces, np.array(IDs))
        recognizer.save("TrainingImageLabel/Trainer.yml")
        message1.config(text="Profile trained successfully!")
    
    except Exception as e:
        mess.showerror('Error', f'Training failed: {str(e)}')

# ------------------------ TRACK IMAGES -----------------------

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainer.yml")
    except Exception:
        mess.showerror('Error', 'Trainer file missing or cv2.face unavailable.')
        return

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    except Exception:
        mess.showerror('Error', 'StudentDetails.csv not found or corrupted!')
        return

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    font = cv2.FONT_HERSHEY_SIMPLEX

    attendance = []
    recorded_ids = set()  # Track already recorded IDs to avoid duplicates
    skip_frames = 3  # Process every 3rd frame
    frame_count = 0
    last_detection_time = defaultdict(float)  # Cooldown for multiple detections
    cooldown_seconds = 5  # Don't record same person within 5 seconds

    mess.showinfo('Instructions', "Press 'Q' to stop attendance capture.")
    
    while True:
        ret, img = cam.read()
        if not ret:
            break

        frame_count += 1

        # Only process every nth frame
        if frame_count % skip_frames == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Optimized detection parameters
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                try:
                    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    
                    if conf < 60:  # Slightly relaxed threshold
                        ts = time.time()
                        
                        # Check cooldown period
                        if ts - last_detection_time[Id] < cooldown_seconds:
                            name = df.loc[df['SERIAL NO.'] == Id]['NAME'].values
                            name = name[0] if len(name) else "Unknown"
                            cv2.putText(img, f"{name} (Recorded)", (x, y - 10), font, 0.7, (0, 255, 0), 2)
                        else:
                            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            name = df.loc[df['SERIAL NO.'] == Id]['NAME'].values
                            ID = df.loc[df['SERIAL NO.'] == Id]['ID'].values
                            name = name[0] if len(name) else "Unknown"
                            ID = ID[0] if len(ID) else "Unknown"
                            
                            attendance.append([ID, name, date, timeStamp])
                            recorded_ids.add(Id)
                            last_detection_time[Id] = ts
                            
                            cv2.putText(img, f"{name} - Marked!", (x, y - 10), font, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, "Unknown", (x, y - 10), font, 0.7, (0, 0, 255), 2)

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                except Exception as e:
                    print(f"Recognition error: {e}")
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Display count
        cv2.putText(img, f"Recorded: {len(attendance)}", (10, 30), font, 0.7, (255, 255, 0), 2)
        cv2.imshow("Taking Attendance (Press Q to stop)", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Save attendance
    if attendance:
        date = datetime.datetime.now().strftime('%d-%m-%Y')
        path = f"Attendance/Attendance_{date}.csv"
        exists = os.path.isfile(path)
        
        with open(path, 'a+', newline='') as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(['ID', 'Name', 'Date', 'Time'])
            writer.writerows(attendance)

        # Update treeview
        for k in tv.get_children():
            tv.delete(k)
        for entry in attendance:
            tv.insert('', 0, text=entry[0], values=(entry[1], entry[2], entry[3]))
        
        mess.showinfo('Success', f'Attendance recorded for {len(attendance)} student(s)!')
    else:
        mess.showinfo('Info', 'No attendance was recorded.')

# -------------------------- GUI -------------------------------

window = tk.Tk()
window.geometry("1280x720")
window.title("Attendance System")
window.configure(bg="#262523")

frame1 = tk.Frame(window, bg="#00aeff")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#00aeff")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="Face Recognition Based Attendance System",
                    fg="white", bg="#262523", font=('times', 28, 'bold'))
message3.place(x=20, y=10)

lbl = tk.Label(frame2, text="Enter ID", fg="black", bg="#00aeff", font=('times', 17, 'bold'))
lbl.place(x=80, y=55)
txt = tk.Entry(frame2, width=32, fg="black", font=('times', 15, 'bold'))
txt.place(x=30, y=88)

lbl2 = tk.Label(frame2, text="Enter Name", fg="black", bg="#00aeff", font=('times', 17, 'bold'))
lbl2.place(x=80, y=140)
txt2 = tk.Entry(frame2, width=32, fg="black", font=('times', 15, 'bold'))
txt2.place(x=30, y=173)

message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile", bg="#00aeff",
                    fg="black", font=('times', 15, 'bold'))
message1.place(x=7, y=230)

lbl3 = tk.Label(frame1, text="Attendance", fg="black", bg="#00aeff", font=('times', 17, 'bold'))
lbl3.place(x=100, y=115)

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')

takeImg = tk.Button(frame2, text="Take Images", command=TakeImages, fg="white", bg="blue",
                    width=34, font=('times', 15, 'bold'))
takeImg.place(x=30, y=300)

trainImg = tk.Button(frame2, text="Save Profile", command=TrainImages, fg="white", bg="blue",
                     width=34, font=('times', 15, 'bold'))
trainImg.place(x=30, y=380)

trackImg = tk.Button(frame1, text="Take Attendance", command=TrackImages, fg="black", bg="yellow",
                     width=35, font=('times', 15, 'bold'))
trackImg.place(x=30, y=50)

quitWindow = tk.Button(frame1, text="Quit", command=window.destroy, fg="black", bg="red",
                       width=35, font=('times', 15, 'bold'))
quitWindow.place(x=30, y=450)

window.mainloop()