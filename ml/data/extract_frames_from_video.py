import cv2
import os
from tqdm import tqdm


def load_frame_count(file_path="value.txt"):
    """Load the current frame count from value.txt file"""
    try:
        with open(file_path, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def save_frame_count(count, file_path="value.txt"):
    """Save the current frame count to value.txt file"""
    with open(file_path, "w") as f:
        f.write(str(count))


def extract_frames(video_path, output_folder, frames_per_second=10):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the current frame count from value.txt
    initial_frame_count = load_frame_count()
    saved_frame_count = initial_frame_count

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get total number of frames and fps
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames to capture
    interval = int(video_fps / frames_per_second)

    frame_count = 0

    # Initialize progress bar
    progress_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        # Update progress bar
        progress_bar.update(1)

        # Save frame every interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(
                output_folder, f"frame_{saved_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

            # Save the current count to value.txt
            save_frame_count(saved_frame_count)

        frame_count += 1

    # Close progress bar
    progress_bar.close()

    # Release the video capture object
    video_capture.release()

    new_frames_extracted = saved_frame_count - initial_frame_count
    print(
        f"Extracted {new_frames_extracted} new frames and saved them to {output_folder}"
    )
    print(f"Total frames saved so far: {saved_frame_count}")


# Example usage
input_video_path = ""
output_picture_folder = ""
extract_frames(input_video_path, output_picture_folder)
