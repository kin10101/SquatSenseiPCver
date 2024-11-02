import cv2
import mediapipe as mp
import pandas as pd
import os
import shutil

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # To draw the pose landmarks
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5)

# Initialize video ID for auto-increment
video_id = 1


# Function to extract pose landmarks from a video frame
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks and len(results.pose_landmarks.landmark) == 33:  # Ensure we have 33 landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return results.pose_landmarks, landmarks  # Return both landmarks and pose results for drawing
    else:
        return None, None


# Function to detect the phase of the squat based on the y-coordinates of the hips and knees
def detect_squat_phase(landmarks):
    left_hip_y = landmarks[23][1]
    right_hip_y = landmarks[24][1]
    left_knee_y = landmarks[25][1]
    right_knee_y = landmarks[26][1]

    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_knee_y = (left_knee_y + right_knee_y) / 2

    if avg_hip_y < avg_knee_y * 0.82:
        return "top"
    elif avg_hip_y < avg_knee_y * 0.88:
        return "middle"
    else:  # Hip is near or below knee level
        return "bottom"


# Function to process video, categorize the phase, prompt for class label, and save data to CSV
def process_video(video_path, csv_path, show_preview, review_dir):
    global video_id
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_landmarks, landmarks = extract_landmarks(frame)
        if landmarks:
            if len(landmarks) == 33:
                phase = detect_squat_phase(landmarks)

                # Flatten the landmarks for each frame
                flattened_landmarks = [coord for point in landmarks[:33] for coord in point]

                # Temporary placeholders; these will be updated with user inputs
                stance, head_position, chest_position, knee_position, heel_position = "", "", "", "", ""

                # Collect the data for each frame
                row = [video_id, stance, phase, head_position, chest_position, knee_position,
                       heel_position] + flattened_landmarks
                data.append(row)

                if show_preview:
                    # Draw the landmarks with a standard color
                    color_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green
                    mp_drawing.draw_landmarks(
                        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=color_spec,
                        connection_drawing_spec=color_spec
                    )

                    cv2.putText(frame, f"ID: {video_id} Phase: {phase}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Display the frame with landmarks drawn
                    cv2.imshow("Pose Estimation", frame)

                    # Exit if 'q' is pressed
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Prompt user to confirm if the data is correct
    confirm = input(f"Is the data for {video_path} correct? (y/n): ").strip().lower()
    if confirm == 'y':
        # Prompt user to enter Stance, Head, Chest, Knee, and Heel values
        stance = input("Enter squat stance (e.g., wide, standard, narrow): ").strip()
        head_position = input("Enter head position (e.g., forward facing, downward facing): ").strip()
        chest_position = input("Enter chest position (e.g., chest up, incorrect): ").strip()
        knee_position = input("Enter knee position (e.g., correct, caving in): ").strip()
        heel_position = input("Enter heel position (e.g., flat, raising): ").strip()

        # Update the values in each row of data
        for row in data:
            row[1] = stance
            row[3] = head_position
            row[4] = chest_position
            row[5] = knee_position
            row[6] = heel_position

        # Define the column names
        columns = ["ID", "Stance", "Phase", "Head", "Chest", "Knee", "Heel"] + \
                  [f"{axis}{i}" for i in range(33) for axis in ["X", "Y", "Z"]]
        df = pd.DataFrame(data, columns=columns)

        # Write to CSV, appending if it exists
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Results saved to {csv_path}")

        # Increment the video ID for the next video
        video_id += 1

    else:
        # Clear the data if not confirmed
        data.clear()

        # Move video to review directory if data is not confirmed as correct
        if not os.path.exists(review_dir):
            os.makedirs(review_dir)
        shutil.move(video_path, os.path.join(review_dir, os.path.basename(video_path)))
        print(f"Video {video_path} moved to review directory: {review_dir}")


# Function to get all video files in a directory
def get_video_files(directory):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(video_extensions)]


# Main function to run the script
if __name__ == "__main__":
    video_directory = r"E:\Pycharm Projects\Squat Sensei\Datasets\Front\clipped\Standard Kin"
    csv_path = r"E:\Pycharm Projects\Squat Sensei\Datasets\kin.csv"
    review_dir = r"E:\Pycharm Projects\Squat Sensei\Datasets\To Review"

    show_preview = True  # Set to False if you don't want to show the video preview

    video_files = get_video_files(video_directory)
    video_files.reverse()

    for video_path in video_files:
        process_video(video_path, csv_path, show_preview, review_dir)
