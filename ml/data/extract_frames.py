import cv2
import os


def extract_frames(video_path, output_folder, frames_per_second=10):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames to capture
    interval = int(video_fps / frames_per_second)

    frame_count = 0
    saved_frame_count = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        # Save frame every interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(
                output_folder, f"frame_{saved_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {saved_frame_count} frames and saved them to {output_folder}")


# Example usage
video_path = "../../data/raw/input_video.mp4"
output_folder = "../../data/processed/extracted_frames"
extract_frames(video_path, output_folder)
