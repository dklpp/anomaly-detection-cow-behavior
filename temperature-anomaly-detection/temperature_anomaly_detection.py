import cv2
import numpy as np
import os
import re
from pathlib import Path
import math

# --- Configuration ---
COLLAGE_DIR = Path("output_sequences")
TIMESTAMPS_FILE = COLLAGE_DIR / "capture_timestamps.txt"
OUTPUT_DIR = Path("./output/temperature_anomalies")
ANOMALY_COLLAGE_PATH = OUTPUT_DIR / "temperature_anomaly_collage.png"

# Anomaly detection thresholds (adjustable)
# A frame is an anomaly if its average temp is 20% higher than the collage's average
AVG_TEMP_THRESHOLD_FACTOR = 1.2
# Or if its max temp is 10% higher than the collage's max
MAX_TEMP_THRESHOLD_FACTOR = 1.1

# --- Setup ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_timestamp_for_frame(trigger_frame_num):
    """Parses the timestamps file to find the time for a given trigger frame."""
    if not TIMESTAMPS_FILE.exists():
        return "Timestamp file not found"
    
    with open(TIMESTAMPS_FILE, 'r') as f:
        for line in f:
            match = re.search(r"time: (\S+) \(Frame: (\d+)\)", line)
            if match:
                time_str, frame_num_str = match.groups()
                if int(frame_num_str) == trigger_frame_num:
                    return time_str
    return "Not Found"

def split_collage(collage_img, grid_size=(16, 16)):
    """Splits a collage image into a grid of individual frames."""
    h, w, _ = collage_img.shape
    frame_h, frame_w = h // grid_size[0], w // grid_size[1]
    frames = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            frame = collage_img[i*frame_h:(i+1)*frame_h, j*frame_w:(j+1)*frame_w]
            frames.append(frame)
    return frames

def create_results_collage(flagged_frames_data, output_path, grid_cols=5):
    """Creates a collage from the flagged anomaly frames."""
    if not flagged_frames_data:
        print("No anomalies found to create a collage.")
        return

    num_frames = len(flagged_frames_data)
    frame_h, frame_w, _ = flagged_frames_data[0]['frame'].shape
    
    grid_rows = math.ceil(num_frames / grid_cols)
    collage_h = grid_rows * frame_h
    collage_w = grid_cols * frame_w
    
    final_collage = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)

    for i, data in enumerate(flagged_frames_data):
        row = i // grid_cols
        col = i % grid_cols
        
        y_offset = row * frame_h
        x_offset = col * frame_w
        
        final_collage[y_offset:y_offset+frame_h, x_offset:x_offset+frame_w] = data['frame']

    cv2.imwrite(str(output_path), final_collage)
    print(f"Anomaly collage saved to: {output_path}")


def main():
    collage_files = sorted(list(COLLAGE_DIR.glob("collage_triggered_at_frame_*.png")))
    print(f"Found {len(collage_files)} collage files to analyze.")

    all_anomalous_frames = []

    for collage_path in collage_files:
        print(f"\nProcessing {collage_path.name}...")
        
        # Extract trigger frame number from filename
        trigger_frame_match = re.search(r"frame_(\d+)\.png", collage_path.name)
        if not trigger_frame_match:
            continue
        trigger_frame = int(trigger_frame_match.group(1))
        
        collage_img = cv2.imread(str(collage_path))
        if collage_img is None:
            print(f"  - Could not read image.")
            continue

        # --- Global (Collage-level) Metrics ---
        collage_hsv = cv2.cvtColor(collage_img, cv2.COLOR_BGR2HSV)
        collage_v_channel = collage_hsv[:, :, 2]
        global_avg_temp = np.mean(collage_v_channel)
        global_max_temp = np.max(collage_v_channel)
        
        print(f"  - Global Metrics: Avg Temp={global_avg_temp:.2f}, Max Temp={global_max_temp}")

        frames = split_collage(collage_img)
        timestamp = get_timestamp_for_frame(trigger_frame)

        # --- Local (Frame-level) Metrics & Anomaly Detection ---
        for i, frame in enumerate(frames):
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_v_channel = frame_hsv[:, :, 2]
            
            local_avg_temp = np.mean(frame_v_channel)
            local_max_temp = np.max(frame_v_channel)

            is_anomaly = False
            if local_avg_temp > global_avg_temp * AVG_TEMP_THRESHOLD_FACTOR:
                is_anomaly = True
            if local_max_temp > global_max_temp * MAX_TEMP_THRESHOLD_FACTOR:
                is_anomaly = True

            if is_anomaly:
                print(f"    - ANOMALY DETECTED in frame {i+1}")
                
                # Add text with metrics to the frame
                info_frame = frame.copy()
                text_ts = f"Time: {timestamp}"
                text_metrics = f"Avg:{local_avg_temp:.1f} (Glo:{global_avg_temp:.1f}) Max:{local_max_temp} (Glo:{global_max_temp})"
                
                cv2.putText(info_frame, text_ts, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(info_frame, text_metrics, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                
                all_anomalous_frames.append({
                    'frame': info_frame,
                    'collage_source': collage_path.name,
                    'frame_index': i,
                    'timestamp': timestamp
                })

    # --- Final Collage Creation ---
    create_results_collage(all_anomalous_frames, ANOMALY_COLLAGE_PATH)

if __name__ == "__main__":
    main()
