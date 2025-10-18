import cv2
import numpy as np
import os
import time
from datetime import timedelta
import argparse


DEFAULT_VIDEO_PATH = "data/20250319_165700.mp4"
DEFAULT_OUTPUT_DIR = "output"

MOTION_PIXEL_THRESHOLD = 5000
CONFIRMATION_FRAMES = 30
INACTIVITY_FRAMES_TO_END_CLIP = 60
PRE_MOTION_PADDING_SECONDS = 5

ROI_X, ROI_Y, ROI_W, ROI_H = 248, 260, 420, 344


def run_stage_1_analysis(video_path, output_dir, test_seconds=None):
    """Perform fast pruning analysis and export cropped motion segments."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cow_pruned_segments.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    FPS = cap.get(cv2.CAP_PROP_FPS)
    if FPS <= 0:
        FPS = 30.0
        print(f"Warning: Could not read FPS. Assuming default = {FPS}")
    PADDING_FRAMES = int(PRE_MOTION_PADDING_SECONDS * FPS)

    # Limit to N seconds for test mode
    frame_limit = int(test_seconds * FPS) if test_seconds else None
    if test_seconds:
        print(f"Test mode: processing only first {test_seconds} seconds ({frame_limit} frames)")

    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, FOURCC, FPS, (ROI_W, ROI_H))
    print(f"Output initialized ({ROI_W}x{ROI_H}): {output_path}")

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    frame_count = 0
    active_motion_start_frame = -1
    inactivity_counter = 0
    motion_segments_log = []
    is_writing_segment = False

    print(f"Starting Stage 1: {video_path} ({FPS:.2f} FPS)")
    start_time_real = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_limit and frame_count > frame_limit:
            print(f"Reached {test_seconds} seconds limit — stopping early.")
            break

        roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
        fgMask = backSub.apply(roi)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        motion_area = np.sum(fgMask == 255)

        # Motion detection
        if motion_area > MOTION_PIXEL_THRESHOLD:
            inactivity_counter = 0
            if active_motion_start_frame == -1:
                start_candidate = max(0, frame_count - PADDING_FRAMES)
                if (frame_count - start_candidate) > CONFIRMATION_FRAMES:
                    active_motion_start_frame = start_candidate
                    is_writing_segment = True
        elif active_motion_start_frame != -1:
            inactivity_counter += 1
            if inactivity_counter >= INACTIVITY_FRAMES_TO_END_CLIP:
                end_frame = frame_count - INACTIVITY_FRAMES_TO_END_CLIP
                start_frame_no_padding = active_motion_start_frame + PADDING_FRAMES
                start_time_str = str(timedelta(seconds=int(start_frame_no_padding / FPS)))
                end_time_str = str(timedelta(seconds=int(end_frame / FPS)))
                duration = (end_frame - active_motion_start_frame) / FPS
                motion_segments_log.append({
                    'clip_start_frame': active_motion_start_frame,
                    'clip_end_frame': end_frame,
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'duration_s': duration
                })
                print(f"Segment: {start_time_str} → {end_time_str} ({duration:.2f}s)")
                active_motion_start_frame = -1
                is_writing_segment = False
                inactivity_counter = 0

        if is_writing_segment:
            cropped_frame = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
            out.write(cropped_frame)

    # Handle active segment at EOF/test stop
    if active_motion_start_frame != -1:
        end_frame = frame_count
        start_frame_no_padding = active_motion_start_frame + PADDING_FRAMES
        start_time_str = str(timedelta(seconds=int(start_frame_no_padding / FPS)))
        end_time_str = str(timedelta(seconds=int(end_frame / FPS)))
        duration = (end_frame - active_motion_start_frame) / FPS
        motion_segments_log.append({
            'clip_start_frame': active_motion_start_frame,
            'clip_end_frame': end_frame,
            'start_time': start_time_str,
            'end_time': end_time_str,
            'duration_s': duration
        })
        print(f"Segment till EOF: {start_time_str} → {end_time_str} ({duration:.2f}s)")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time_real = time.time()
    total_runtime = end_time_real - start_time_real
    total_video_s = frame_count / FPS

    print("\n--- SUMMARY ---")
    print(f"Processed frames: {frame_count}")
    print(f"Video duration analyzed: {str(timedelta(seconds=int(total_video_s)))}")
    print(f"Processing speed: {frame_count / total_runtime:.2f} FPS")
    print(f"Motion segments detected: {len(motion_segments_log)}")
    total_clip_time = sum(s['duration_s'] for s in motion_segments_log)
    print(f"Total motion time: {total_clip_time:.2f}s")
    print(f"Output saved to: {os.path.abspath(output_path)}")

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run motion-based pruning on a video file.")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH,
                        help="Path to the input MP4 video.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save the output video.")
    parser.add_argument("--test-seconds", type=int, default=None,
                        help="If set, process only the first N seconds of the video (for testing).")

    args = parser.parse_args()

    run_stage_1_analysis(
        video_path=args.video,
        output_dir=args.output_dir,
        test_seconds=args.test_seconds
    )
