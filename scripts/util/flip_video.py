import cv2
import sys

def flip_video_180(input_path, output_path):
    """
    Flips a video 180 degrees (both horizontally and vertically).

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the flipped video file.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object to save the output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        cap.release()
        return

    print(f"Processing video: {input_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Flip the frame 180 degrees (flip around both x and y axes)
        flipped_frame = cv2.flip(frame, -1)

        # Write the flipped frame to the output file
        out.write(flipped_frame)

    # Release everything
    cap.release()
    out.release()
    print(f"Flipped video saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python flip_video.py <input_video.mp4> <output_video.mp4>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]
    flip_video_180(input_video, output_video)
