import subprocess
import os

def compress_video(input_path, output_path, target_fps=30):
    # Construct the ffmpeg command
    command = [
        'ffmpeg',
        '-i', input_path,  # Input file
        '-r', str(target_fps),  # Set the frame rate
        '-vf', 'scale=-1:480',  # Resize to 480p
        '-vcodec', 'libx264',  # Video codec
        '-crf', '28',  # Constant Rate Factor (lower value means better quality)
        '-preset', 'fast',  # Preset for compression speed
        output_path  # Output file
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Compressed video saved to {output_path}")
        print(f"Original video size: {os.path.getsize(input_path)} bytes")
        print(f"Compressed video size: {os.path.getsize(output_path)} bytes")
    except subprocess.CalledProcessError as e:
        print(f"Error during compression: {e}")


def compress_all_videos(input_dir, output_dir):
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)
        compress_video(input_path, output_path)

if __name__ == "__main__":
    input_video_path = r'/Datasets/Front/clipped'  # Replace with your input video path
    output_video_path = r'/Datasets/Front/compressed'  # Replace with your desired output video path

    compress_all_videos(input_video_path, output_video_path)