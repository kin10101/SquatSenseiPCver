import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Helper function to calculate the average y-coordinate of hips
def get_hip_y(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    return (left_hip.y + right_hip.y) / 2

# Video processing function
def process_video(video_path, output_dir, threshold=0.02, min_top_duration=15):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error opening video file")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_rep = 1
    recording = False
    top_frames_count = 0
    previous_y = None
    video_writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            hip_y = get_hip_y(results.pose_landmarks.landmark)

            # Initialize previous_y on the first frame
            if previous_y is None:
                previous_y = hip_y

            # Detect if the person is in the top phase (based on small change in hip y-coordinate)
            if abs(hip_y - previous_y) < threshold:
                top_frames_count += 1
                # If enough frames have passed in the top phase, start a new rep recording
                if top_frames_count >= min_top_duration and not recording:
                    recording = True
                    video_writer = cv2.VideoWriter(
                        os.path.join(output_dir, f"rep_{current_rep}.mp4"),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (frame_width, frame_height)
                    )
                    current_rep += 1
                    print("Starting new rep recording...")
            else:
                # Reset top_frames_count when the y-coordinate changes significantly (moving down)
                top_frames_count = 0

            # If in recording mode, write the current frame to the video
            if recording:
                video_writer.write(frame)

            # Detect end of rep when the person returns to the top position
            if recording and top_frames_count >= min_top_duration:
                recording = False
                video_writer.release()
                video_writer = None
                print("Rep completed and saved.")

            # Update the previous_y
            previous_y = hip_y

    cap.release()
    pose.close()
    if video_writer:
        video_writer.release()
    print("All reps have been processed and saved.")

# Example usage
video_path = r"E:\Pycharm Projects\Squat Sensei\Datasets\Front\compressed\eron.mp4"
output_dir = r"E:\Pycharm Projects\Squat Sensei\Datasets\Front\compressed\segmented"
process_video(video_path, output_dir)
