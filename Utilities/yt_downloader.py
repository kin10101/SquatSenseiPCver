import subprocess

def download_video(url, output_path):
    # Construct the yt-dlp command
    command = [
        'yt-dlp',
        '-o', output_path,  # Output path template
        url  # URL of the video to download
    ]

    # Run the yt-dlp command
    try:
        subprocess.run(command, check=True)
        print(f"Video downloaded successfully to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    folder_path = r'/Datasets/Front'  # Replace with your desired output folder path
    video_url = 'https://www.youtube.com/watch?v=KCSYgpcZVPs'  # Replace with the actual video URL
    output_template = 'downloaded_video.mp4'  # Replace with your desired output path template

    download_video(video_url, output_template)