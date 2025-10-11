# Python Script for Two-Stage Video Filtering: Stage 1 - Fast Pruning
# This script uses Background Subtraction (MOG2) to quickly scan a long video
# for 10-second segments where a cow is entering or exiting the milking area,
# and EXPORTS THE PRUNED SEGMENTS cropped to the ROI to a new video file.

import cv2
import numpy as np
import time
from datetime import timedelta
import os 

VIDEO_PATH = "anomaly-detection-cow-behavior/data/cow_full.mp4" 
OUTPUT_VIDEO_PATH = "cow_pruned_segments_full.mp4"

# PIXEL THRESHOLD: Foreground pixels needed to register significant motion
MOTION_PIXEL_THRESHOLD = 5000 

# CONFIRMATION FRAMES: How many consecutive frames must exceed the threshold to confirm motion started.
CONFIRMATION_FRAMES = 30 

# INACTIVITY TIME: How long (in frames) the motion must be absent to confirm the cow has left the critical zone.
INACTIVITY_FRAMES_TO_END_CLIP = 30

# WINDOW PADDING: How many seconds of video to include *before* the motion starts.
PRE_MOTION_PADDING_SECONDS = 2
PADDING_FRAMES = 0 

# Region of Interest (ROI) 
ROI_X, ROI_Y, ROI_W, ROI_H = 248, 260, 420, 344

def run_stage_1_analysis():
    """
    The main function to perform the fast pruning analysis and generate the output video.
    """
    if ROI_W <= 0 or ROI_H <= 0:
        print("Error: ROI dimensions are not valid.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    # Video properties
    # FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # No longer needed for output size
    # FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # No longer needed for output size
    FPS = cap.get(cv2.CAP_PROP_FPS)
    
    if FPS <= 0:
        FPS = 30.0
        print(f"Warning: Could not read FPS. Assuming default: {FPS}")

    global PADDING_FRAMES
    PADDING_FRAMES = int(PRE_MOTION_PADDING_SECONDS * FPS)

    # --- Video Writer Initialization (MODIFIED) ---
    # The VideoWriter now uses the ROI width and height, not the full frame size.
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, FOURCC, FPS, (ROI_W, ROI_H))
    print(f"Output video initialized with cropped dimensions ({ROI_W}x{ROI_H}): {OUTPUT_VIDEO_PATH}")

    # Initialize Background Subtractor (MOG2)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # State tracking variables
    frame_count = 0
    active_motion_start_frame = -1
    inactivity_counter = 0
    motion_segments_log = []
    is_writing_segment = False 

    print(f"Starting Stage 1: Scanning {VIDEO_PATH} at {FPS:.2f} FPS...")
    start_time_real = time.time()

    # Pruning and Writing Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1

        # Apply ROI and Background Subtraction
        # Note: The 'roi' variable is only used for motion detection, not for output.
        roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
        fgMask = backSub.apply(roi)

        # Filter out non-colored (grayscale) pixels from the motion mask ---
        # A pixel is considered non-colored if the standard deviation of its B,G,R channels is low.
        # We calculate the standard deviation across the color channels for each pixel in the ROI.
        std_dev_channels = np.std(roi, axis=2)
        
        # Create a mask where pixels are considered 'colored' if their channel std dev is above a threshold.
        # This threshold can be tuned. A lower value is more sensitive to color.
        COLOR_SENSITIVITY_THRESHOLD = 10 
        color_mask = (std_dev_channels > COLOR_SENSITIVITY_THRESHOLD).astype(np.uint8) * 255

        # The fgMask from the background subtractor contains shadows (value 127). We only want definite foreground (value 255).
        foreground_only_mask = (fgMask == 255).astype(np.uint8) * 255
        
        # Combine the two masks: we want pixels that are BOTH foreground AND colored.
        colored_foreground_mask = cv2.bitwise_and(foreground_only_mask, color_mask)

        # Clean up mask 
        kernel = np.ones((5, 5), np.uint8)
        # The morphology is now applied to the new, filtered mask.
        cleaned_mask = cv2.morphologyEx(colored_foreground_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion area from the cleaned, color-filtered mask
        motion_area = np.sum(cleaned_mask == 255)

        # --- Motion Detection Logic ---

        if motion_area > MOTION_PIXEL_THRESHOLD:
            # Motion detected: Reset inactivity counter
            inactivity_counter = 0
            
            if active_motion_start_frame != -1 and not is_writing_segment:
                is_writing_segment = True
            
            if active_motion_start_frame == -1:
                # Start the event window with padding (cannot go before frame 0)
                start_frame_candidate = max(0, frame_count - PADDING_FRAMES)
                
                # Use a confirmation buffer to avoid logging flicker/noise
                if (frame_count - start_frame_candidate) > CONFIRMATION_FRAMES:
                    active_motion_start_frame = start_frame_candidate
                    is_writing_segment = True # Start writing immediately after confirmation

        elif active_motion_start_frame != -1:
            # Motion has stopped, but we are inside an active segment
            inactivity_counter += 1

            if inactivity_counter >= INACTIVITY_FRAMES_TO_END_CLIP:
                # Inactivity threshold reached: The cow has left the zone.
                
                # Define the end of the segment (adjusted backward by the buffer)
                end_frame = frame_count - INACTIVITY_FRAMES_TO_END_CLIP
                
                # Log the full segment
                start_frame_no_padding = active_motion_start_frame + PADDING_FRAMES
                
                # Convert frames to timestamps
                start_time_str = str(timedelta(seconds=int(start_frame_no_padding / FPS)))
                end_time_str = str(timedelta(seconds=int(end_frame / FPS)))

                segment = {
                    'clip_start_frame': active_motion_start_frame,
                    'clip_end_frame': end_frame,
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'duration_s': (end_frame - active_motion_start_frame) / FPS
                }
                motion_segments_log.append(segment)
                
                print(f"Segment found: Start={start_time_str} (Frame {active_motion_start_frame}) | Duration={segment['duration_s']:.2f}s")

                is_writing_segment = False 
                active_motion_start_frame = -1
                inactivity_counter = 0

        # --- Write Frame Logic (MODIFIED) ---
        if is_writing_segment:
            # Crop the frame to the ROI before writing to the output video
            cropped_frame = frame[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
            out.write(cropped_frame)
            
    # --- Cleanup and Finalize Video File ---
    
    # Check if a segment was active when the video ended (e.g., cow is still in the zone)
    if active_motion_start_frame != -1:
        end_frame = frame_count 
        start_frame_no_padding = active_motion_start_frame + PADDING_FRAMES
        
        start_time_str = str(timedelta(seconds=int(start_frame_no_padding / FPS)))
        end_time_str = str(timedelta(seconds=int(end_frame / FPS)))
        
        segment = {
            'clip_start_frame': active_motion_start_frame,
            'clip_end_frame': end_frame,
            'start_time': start_time_str,
            'end_time': end_time_str,
            'duration_s': (end_frame - active_motion_start_frame) / FPS
        }
        motion_segments_log.append(segment)
        print(f"Segment found (Ended by EOF): Start={start_time_str} (Frame {active_motion_start_frame}) | Duration={segment['duration_s']:.2f}s")


    cap.release()
    out.release() 
    cv2.destroyAllWindows()

    end_time_real = time.time()
    total_runtime_real = end_time_real - start_time_real
    total_video_duration_s = frame_count / FPS
    total_video_duration_str = str(timedelta(seconds=int(total_video_duration_s)))

    print("\n--- STAGE 1 SUMMARY ---")
    print(f"Video Duration: {total_video_duration_str}")
    print(f"Total Processing Time: {total_runtime_real:.2f} seconds")
    print(f"Processing Speed: {frame_count / total_runtime_real:.2f} FPS")
    print(f"Detected {len(motion_segments_log)} motion segments.")

    total_clip_duration_s = sum(s['duration_s'] for s in motion_segments_log)
    print(f"Total clip time for Stage 2: {total_clip_duration_s / 60:.2f} minutes")
    print(f"Original video time saved: {(total_video_duration_s - total_clip_duration_s) / 3600:.2f} hours")
    print(f"Generated output video (Cropped to ROI): {os.path.abspath(OUTPUT_VIDEO_PATH)}")

    print("\n--- MOTION SEGMENTS (for Stage 2 Analysis) ---")
    for i, segment in enumerate(motion_segments_log):
        print(f"Clip {i+1}: Frames {segment['clip_start_frame']} to {segment['clip_end_frame']} | Time {segment['start_time']} to {segment['end_time']}")

run_stage_1_analysis()
