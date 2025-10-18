import cv2
import numpy as np
import sys
import os
import math
import datetime
import datetime

def count_pixels_in_frame(frame):
    """Counts colored pixels in a single video frame."""
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([35, 50, 50])
    upper_bound = np.array([130, 255, 255])
    colored_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return cv2.countNonZero(colored_mask)

def create_fusion(frames, output_path):
    """Creates a single image by averaging a list of frames."""
    fused_image = np.mean(frames, axis=0).astype(np.uint8)
    cv2.imwrite(output_path, fused_image)
    print(f"Saved fusion image to: {output_path}")

def create_collage(frames, output_path):
    """Creates a collage from a list of frames."""
    num_frames = len(frames)
    if num_frames == 0:
        return

    frame_h, frame_w, _ = frames[0].shape
    cols = int(math.ceil(math.sqrt(num_frames)))
    rows = int(math.ceil(num_frames / cols))

    # Pad the list with black frames if it doesn't fit the grid perfectly
    padding_needed = rows * cols - num_frames
    padded_frames = frames + [np.zeros_like(frames[0]) for _ in range(padding_needed)]

    # Create rows of images
    img_rows = []
    for i in range(rows):
        start = i * cols
        end = start + cols
        img_row = cv2.hconcat(padded_frames[start:end])
        img_rows.append(img_row)

    # Combine rows into the final collage
    collage = cv2.vconcat(img_rows)
    cv2.imwrite(output_path, collage)
    print(f"Saved collage image to: {output_path}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path} @ {fps:.2f} FPS")

    frame_idx = 0
    pixel_threshold = 45000
    capture_duration_seconds = 10
    frames_to_capture_count = int(fps * capture_duration_seconds)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_idx % 5 == 0:
            pixel_count = count_pixels_in_frame(frame)

            if pixel_count > pixel_threshold:
                print(f"TRIGGER at frame {frame_idx} with {pixel_count} pixels.")

                # Define output directory and log the trigger timestamp
                output_dir = "output_sequences"
                os.makedirs(output_dir, exist_ok=True)

                timestamp_seconds = frame_idx / fps
                formatted_timestamp = str(datetime.timedelta(seconds=timestamp_seconds))
                log_filename = os.path.join(output_dir, "capture_timestamps.txt")
                log_line = f"Capture triggered at video time: {formatted_timestamp} (Frame: {frame_idx})\n"
                with open(log_filename, 'a') as log_file:
                    log_file.write(log_line)
                print(f"Logged trigger event to {log_filename}")
                
                collected_frames = [frame]
                start_frame_of_sequence = frame_idx

                for i in range(frames_to_capture_count - 1):
                    ret_inner, next_frame = cap.read()
                    if not ret_inner:
                        break
                    collected_frames.append(next_frame)
                    frame_idx += 1
                
                print(f"Collected {len(collected_frames)} frames for processing.")
                
                fusion_path = os.path.join(output_dir, f"fusion_triggered_at_frame_{start_frame_of_sequence}.png")
                collage_path = os.path.join(output_dir, f"collage_triggered_at_frame_{start_frame_of_sequence}.png")

                create_fusion(collected_frames, fusion_path)
                create_collage(collected_frames, collage_path)

                # Cooldown: Skip ahead 1 minute after the capture sequence
                skip_duration_seconds = 60*10
                frames_to_skip = int(fps * skip_duration_seconds)
                target_frame = frame_idx + frames_to_skip
                
                print(f"Cooldown: Skipping {skip_duration_seconds} seconds to frame ~{target_frame}.")
                
                # Set the video capture to the new position.
                # We set it to target_frame - 1 because the main loop's cap.read() and frame_idx increment
                # will land it on the correct target_frame for the next iteration.
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
                
                # Update the main frame counter to match
                frame_idx = target_frame - 1

        frame_idx += 1

    cap.release()
    print("Video processing complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_thermal_video.py <path_to_video>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    process_video(input_path)