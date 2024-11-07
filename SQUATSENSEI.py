import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import warnings
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from playsound import playsound
from PIL import Image, ImageTk

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Global variables
rep_count = 0
last_phase = None
expected_phase_sequence = ["top", "middle", "bottom", "middle", "top"]
current_phase_sequence = []
frame_counts = {"head": 0, "chest": 0, "knee": 0, "heel": 0}
errors = {"head": 1, "chest": 1, "knee": 1, "heel": 1}  # Start with correct form
threshold = 5  # Threshold for errors in consecutive frames



# Load pre-trained models
with open('Models/stance_model.pkl', 'rb') as file:
    stance_model = pickle.load(file)
with open('Models/phase_model.pkl', 'rb') as file:
    phase_model = pickle.load(file)
with open('Models/head_model.pkl', 'rb') as file:
    head_model = pickle.load(file)
with open('Models/chest_model.pkl', 'rb') as file:
    chest_model = pickle.load(file)
with open('Models/heel_model.pkl', 'rb') as file:
    heel_model = pickle.load(file)
with open('Models/knee_model.pkl', 'rb') as file:  # Load knee model
    knee_model = pickle.load(file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5)

# Function to reset frame counts and errors after each rep
def reset_error_tracking():
    global frame_counts, errors
    frame_counts = {"head": 0, "chest": 0, "knee": 0, "heel": 0}
    errors = {"head": 1, "chest": 1, "knee": 1, "heel": 1}





def count_reps(phase):
    global rep_count, last_phase, current_phase_sequence

    # Check if we need to add a new phase to the sequence
    if last_phase != phase:
        last_phase = phase
        current_phase_sequence.append(phase)

        # Only keep the last 5 phases to check for a complete rep
        if len(current_phase_sequence) > len(expected_phase_sequence):
            current_phase_sequence.pop(0)

        # Check if the current sequence matches the expected sequence
        if current_phase_sequence == expected_phase_sequence:
            rep_count += 1
            current_phase_sequence = []  # Reset for the next rep

            # Play the audio feedback based on errors
            error_code = "".join(str(v) for v in errors.values())
            audio_path = "E:\\Pycharm Projects\\Squat Sensei\\Feedback\\" + f"{error_code}.mp3"
            playsound(audio_path)
            reset_error_tracking()  # Reset error tracking for the next rep


# Function to extract pose landmarks from a video frame
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks and len(results.pose_landmarks.landmark) == 33:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return results.pose_landmarks, landmarks  # Return both landmarks and pose results for drawing
    return None, None


# Function to classify based on extracted landmarks
def classify_pose(landmarks):
    global frame_counts, errors

    # Extract subsets of landmarks based on the training configuration
    stance_landmarks = [landmarks[i] for i in [23, 24, 31, 32]]
    phase_landmarks = [landmarks[i] for i in [11, 12, 23, 24, 25, 26, 27, 28]]
    head_landmarks = [landmarks[i] for i in [0, 1, 2, 4, 5, 11, 12]]
    chest_landmarks = [landmarks[i] for i in [11, 12, 23, 24]]
    heel_landmarks = [landmarks[i] for i in [27, 28, 29, 30, 31, 32]]
    knee_landmarks = [landmarks[i] for i in [23, 24, 25, 26, 31, 32]]

    # Flatten each subset of landmarks
    stance_flat = [coord for point in stance_landmarks for coord in point]
    phase_flat = [coord for point in phase_landmarks for coord in point]
    head_flat = [coord for point in head_landmarks for coord in point]
    chest_flat = [coord for point in chest_landmarks for coord in point]
    heel_flat = [coord for point in heel_landmarks for coord in point]
    knee_flat = [coord for point in knee_landmarks for coord in point]

    # Create DataFrame with feature names for each subset
    stance_df = pd.DataFrame([stance_flat], columns=stance_model.feature_names_in_)
    phase_df = pd.DataFrame([phase_flat], columns=phase_model.feature_names_in_)
    head_df = pd.DataFrame([head_flat], columns=head_model.feature_names_in_)
    chest_df = pd.DataFrame([chest_flat], columns=chest_model.feature_names_in_)
    heel_df = pd.DataFrame([heel_flat], columns=heel_model.feature_names_in_)
    knee_df = pd.DataFrame([knee_flat], columns=knee_model.feature_names_in_)

    # Make predictions using the loaded models with the corresponding flattened landmarks
    stance_pred = stance_model.predict(stance_df)[0]
    phase_pred = phase_model.predict(phase_df)[0]
    head_pred = head_model.predict(head_df)[0]
    chest_pred = chest_model.predict(chest_df)[0]
    knee_pred = knee_model.predict(knee_df)[0]
    heel_pred = heel_model.predict(heel_df)[0]

    # Initialize predictions dictionary
    predictions = {
        "stance": stance_pred,
        "phase": app.phase_detector(landmarks), #"phase": phase_pred,
        "head": "facing forwards" if head_pred > 0 else "facing downwards",
        "chest": "chest up" if chest_pred > 0 else "chest down",
        "knee": "correct position" if knee_pred > 0 else "knees collapsing",
        "heel": "heels flat" if heel_pred > 0 else "heels raising"
    }

    # Update error counts based on predictions
    for key in ["head", "chest", "knee", "heel"]:
        if predictions[key] in ["facing downwards", "chest down", "knees collapsing", "heels raising"]:
            frame_counts[key] += 1
            # If the error count meets threshold, update errors for feedback
            if frame_counts[key] >= threshold:
                errors[key] = 0  # Set to error state if threshold met
        else:
            frame_counts[key] = 0  # Reset count if corrected

    return predictions


# GUI application class
class SquatSenseiApp:
    def __init__(self, root):
        self.calibrated = False
        self.root = root
        self.root.title("Squat Sensei")

        # Get the screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate window dimensions (80% of screen size)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        # Set the size of the window
        self.root.geometry(f"{window_width}x{window_height}")

        # Center the window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"+{x}+{y}")

        # Frame for camera feed
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Label for displaying the video
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Buttons to start live classification, evaluate video, and exit
        self.start_button = tk.Button(root, text="Start Live Classification", command=self.start_live_classification)
        self.start_button.pack(pady=10)

        self.calibrate_button = tk.Button(root, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack()

        self.exit_button = tk.Button(root, text="Exit", command=root.quit)
        self.exit_button.pack(pady=10)

        # Start the camera feed
        self.cap = cv2.VideoCapture(0)  # Change to video file path if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_video()

    def phase_detector(self, landmarks):
        left_hip_y = landmarks[23][1]
        right_hip_y = landmarks[24][1]
        left_knee_y = landmarks[25][1]
        right_knee_y = landmarks[26][1]

        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_knee_y = (left_knee_y + right_knee_y) / 2

        if avg_hip_y < avg_knee_y * .84:  # Hip is much higher than the knee
            return "top"
        elif avg_hip_y < avg_knee_y * 0.88:  # Hip is getting close to knee level
            return "middle"
        else:  # Hip is near or below knee level
            return "bottom"

    def calibrate(self):
        # Dummy calibration function; replace with your calibration logic
        if not self.calibrated:
            self.calibrated = True
            messagebox.showinfo("Calibration", "Calibration complete!")
        else:
            messagebox.showinfo("Calibration", "Already calibrated.")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Image
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            # Update the video label with the new image
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        # Call this method again after 10 ms
        self.root.after(10, self.update_video)

    def start_live_classification(self):
        # Start the live classification in a new thread
        threading.Thread(target=self.live_classification).start()

    def live_classification(self, ip_camera_url=0):
        print("Starting live classification...")
        try:
            cap = cv2.VideoCapture("http://192.168.100.54:8080/video")
        except:
            cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract landmarks from the frame
            pose_landmarks, landmarks = extract_landmarks(frame)
            if landmarks:
                # Classify the current pose
                predictions = classify_pose(landmarks)
                count_reps(predictions["phase"])

                # Set color based on phase prediction
                if predictions['phase'] == 'top':
                    color_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                elif predictions['phase'] == 'middle':
                    color_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                elif predictions['phase'] == 'bottom':
                    color_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                else:
                    color_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=color_spec,
                    connection_drawing_spec=color_spec,
                )

                text_color = (128, 0, 128)
                # Display predictions on the frame
                cv2.putText(frame, f"Stance: {predictions['stance']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            text_color,
                            2)
                cv2.putText(frame, f"Phase: {predictions['phase']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            text_color, 2)
                cv2.putText(frame, f"Head Position: {predictions['head']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            text_color, 2)
                cv2.putText(frame, f"Chest Position: {predictions['chest']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            text_color, 2)
                cv2.putText(frame, f"Knee Position: {predictions['knee']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            text_color, 2)
                cv2.putText(frame, f"Heel Position: {predictions['heel']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            text_color, 2)
                cv2.putText(frame, f"Rep Count: {rep_count}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                # Convert frame to RGB for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the video label with the new image
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Call this method again after a brief delay to process the next frame
            self.root.after(10, self.update_video)



    def __del__(self):
        # Release the camera when the app is closed
        if self.cap.isOpened():
            self.cap.release()


# Create the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = SquatSenseiApp(root)
    root.mainloop()
