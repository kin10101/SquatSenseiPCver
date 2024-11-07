import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import warnings
import keras

from playsound import playsound

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

rep_count = 0
last_phase = None
expected_phase_sequence = ["top", "middle", "bottom", "middle", "top"]
current_phase_sequence = []
frame_counts = {"head": 0, "chest": 0, "knee": 0, "heel": 0}
errors = {"head": 1, "chest": 1, "knee": 1, "heel": 1}  # Start with correct form
threshold = 5  # Threshold for errors in consecutive frames

stance_model = keras.saving.load_model("Models/stance_model.keras")
phase_model = keras.saving.load_model("Models/phase_model.keras")
head_model = keras.saving.load_model("Models/head_model.keras")
chest_model = keras.saving.load_model("Models/chest_model.keras")
heel_model = keras.saving.load_model("Models/heel_model.keras")
knee_model = keras.saving.load_model("Models/knee_model.keras")

print("Models loaded successfully")

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


def phase_detector(landmarks):
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

    # Define feature names (make sure this matches the input shape of your model)
    stance_features = [f'feature_{i}' for i in range(len(stance_flat))]
    phase_features = [f'feature_{i}' for i in range(len(phase_flat))]
    head_features = [f'feature_{i}' for i in range(len(head_flat))]
    chest_features = [f'feature_{i}' for i in range(len(chest_flat))]
    heel_features = [f'feature_{i}' for i in range(len(heel_flat))]
    knee_features = [f'feature_{i}' for i in range(len(knee_flat))]

    # Create DataFrame with feature names for each subset
    stance_df = pd.DataFrame([stance_flat], columns=stance_features)
    phase_df = pd.DataFrame([phase_flat], columns=phase_features)
    head_df = pd.DataFrame([head_flat], columns=head_features)
    chest_df = pd.DataFrame([chest_flat], columns=chest_features)
    heel_df = pd.DataFrame([heel_flat], columns=heel_features)
    knee_df = pd.DataFrame([knee_flat], columns=knee_features)

    # Make predictions using the loaded models with the corresponding flattened landmarks
    stance_pred = stance_model.predict(stance_df)[0]
    phase_pred = phase_model.predict(phase_df)[0]
    head_pred = head_model.predict(head_df)[0]
    chest_pred = chest_model.predict(chest_df)[0]
    knee_pred = knee_model.predict(knee_df)[0]
    heel_pred = heel_model.predict(heel_df)[0]

    print(stance_pred, phase_pred, head_pred, chest_pred, knee_pred, heel_pred)

    # Initialize predictions dictionary
    predictions = {
        "stance": stance_pred,
        "phase": phase_detector(landmarks),  # "phase": phase_pred,
        "head": "facing forwards" if head_pred.argmax() > 0 else "facing downwards",
        "chest": "chest up" if chest_pred.argmax() > 0 else "chest down",
        "knee": "correct position" if knee_pred.argmax() > 0 else "knees collapsing",
        "heel": "heels flat" if heel_pred.argmax() > 0 else "heels raising"
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


# Main function to capture video from the camera and classify in real-time

# Main function to capture video from the camera and classify in real-time
def live_classification(ip_camera_url):
    try:
        cap = cv2.VideoCapture(ip_camera_url)
    except Exception as e:
        print("Error opening video source")
        cap = cv2.VideoCapture(0)

    # Set the resolution to 16:9 aspect ratio
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

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
            cv2.putText(frame, f"Stance: {predictions['stance']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color,
                        2)
            cv2.putText(frame, f"Phase: {predictions['phase']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Head Position: {predictions['head']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Chest Position: {predictions['chest']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Knee Position: {predictions['knee']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Heel Position: {predictions['heel']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Rep Count: {rep_count}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


        # Display the frame with predictions
        cv2.imshow("Pose Classification", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to evaluate a video file with rep counting
def evaluate_video(video_path):
    cap = cv2.VideoCapture(video_path)

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
            print(current_phase_sequence)

            # Set color based on phase prediction
            if predictions['phase'] == 'top':
                color_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            elif predictions['phase'] == 'middle':
                color_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            elif predictions['phase'] == 'bottom':
                color_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            else:
                color_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(
                frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=color_spec,
                connection_drawing_spec=color_spec
            )

            text_color = (255, 255, 0)
            # Display predictions and rep count on the frame
            cv2.putText(frame, f"Stance: {predictions['stance']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color,
                        2)
            cv2.putText(frame, f"Phase: {predictions['phase']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Head Position: {predictions['head']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Chest Position: {predictions['chest']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Knee Position: {predictions['knee']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Heel Position: {predictions['heel']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        text_color, 2)
            cv2.putText(frame, f"Rep Count: {rep_count}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Display the frame with predictions and rep count
        cv2.imshow("Pose Classification", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the live classification
if __name__ == "__main__":
    live_classification(ip_camera_url="http://192.168.100.54:5000/video")
    # path = r"E:\Pycharm Projects\Squat Sensei\Datasets\Front\compressed\kin"
    # video_files = [f for f in os.listdir(path) if f.endswith('.mp4')]
    # for video_file in video_files:
    #     evaluate_video(os.path.join(path, video_file))
