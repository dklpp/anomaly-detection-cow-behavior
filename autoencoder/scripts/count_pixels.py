import cv2
import numpy as np
import sys
import os
import argparse
from typing import Optional, List, Tuple

# --- 1. Matplotlib Backend Fix (For Pop-up Windows) ---
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    pass # Fallback to default if TkAgg isn't found
import matplotlib.pyplot as plt

# --- 2. ROI Helper Class ---
class ROI:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
    
    @property
    def slice(self) -> Tuple[slice, slice]:
        return (slice(self.y, self.y + self.h), slice(self.x, self.x + self.w))

    def __repr__(self):
        return f"ROI(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

# --- 3. Optimized Activity Function ---
def compute_activity(
    cap: cv2.VideoCapture,
    roi: Optional[ROI],
    mode: str,
    bright_threshold: int,
    color_sat_threshold: int,
    color_value_threshold: int,
    frame_stride: int = 2,
    cow_pixel_threshold: int = 150000,
) -> np.ndarray:
    
    counts: List[int] = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    # State tracking
    cow_in_frame = False
    max_pixels_found = 0
    last_event_time = -1.0 # Initialize to negative so events can trigger at 0.0s

    # Pre-calculate bounds
    lower_color_bound = np.array([0, color_sat_threshold, color_value_threshold])
    upper_color_bound = np.array([180, 255, 255])
    
    print(f"Processing {total_frames} frames (Step: Every {frame_stride})...")
    print(f"Event Threshold: {cow_pixel_threshold} pixels")
    if roi:
        print(f"Applied ROI: {roi}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Handle ROI
        if roi is not None:
            roi_slice_y, roi_slice_x = roi.slice
            if (roi.y + roi.h > frame.shape[0]) or (roi.x + roi.w > frame.shape[1]):
                region = frame[roi.y:min(roi.y+roi.h, frame.shape[0]), 
                               roi.x:min(roi.x+roi.w, frame.shape[1])]
            else:
                region = frame[roi_slice_y, roi_slice_x]
        else:
            region = frame

        # Processing
        current_count = 0
        if mode == "green":
            green_channel = region[:, :, 1]
            _, mask = cv2.threshold(green_channel, bright_threshold, 255, cv2.THRESH_BINARY)
            current_count = cv2.countNonZero(mask)
        elif mode == "color":
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_color_bound, upper_color_bound)
            current_count = cv2.countNonZero(mask)
        else:
            raise ValueError(f"Unsupported detection mode: {mode}")

        counts.append(current_count)
        
        # Track max for debugging
        if current_count > max_pixels_found:
            max_pixels_found = current_count

        # --- COW ENTRY/EXIT LOGIC (With 1s Tolerance) ---
        time_sec = frame_idx / fps if fps > 0 else 0

        # Check Entry
        if not cow_in_frame and current_count > cow_pixel_threshold:
            # Only trigger if 1 second has passed since the last event (or if it's the first event)
            if (time_sec - last_event_time) > 1.0:
                cow_in_frame = True
                last_event_time = time_sec
                sys.stdout.write("\033[K") 
                print(f"\n[EVENT] Cow ENTERED at Frame {frame_idx} ({time_sec:.2f}s) | Pixels: {current_count}")
        
        # Check Exit
        elif cow_in_frame and current_count < cow_pixel_threshold:
            # Only trigger if 1 second has passed since the last event
            if (time_sec - last_event_time) > 1.0:
                cow_in_frame = False
                last_event_time = time_sec
                sys.stdout.write("\033[K") 
                print(f"\n[EVENT] Cow EXITED at Frame {frame_idx} ({time_sec:.2f}s) | Pixels: {current_count}")

        frame_idx += 1
        
        # UPDATED Progress Indicator: Shows current pixel count live
        if frame_idx % 20 == 0: # Update more frequently
            sys.stdout.write(f"\rScanning: {frame_idx}/{total_frames} | Current Pixels: {current_count}")
            sys.stdout.flush()

    print("\n\n--- SUMMARY ---")
    print(f"Processing complete.")
    print(f"Max Pixels Detected: {max_pixels_found}")
    print(f"Threshold used:      {cow_pixel_threshold}")
    if max_pixels_found < cow_pixel_threshold:
        print("(!) WARNING: The max pixels detected never reached your threshold.")
        print("    Try running again with: --cow-threshold " + str(int(max_pixels_found * 0.8)))
    
    return np.asarray(counts, dtype=np.float32)

# --- 4. Plotting Function ---
def plot_results(data: np.ndarray, mode: str, video_name: str, threshold: int):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_axis = np.arange(len(data))
    plot_color = '#00FF00' if mode == 'green' else '#FF00AA' 
    
    ax.plot(x_axis, data, color=plot_color, linewidth=1, alpha=0.9)
    ax.fill_between(x_axis, data, color=plot_color, alpha=0.4)
    
    # Add threshold line
    ax.axhline(y=threshold, color='w', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
    ax.legend()

    ax.set_title(f"Activity Diagram: {video_name} ({mode})")
    ax.set_xlabel("Time (sampled frames)")
    ax.set_ylabel("Pixel Volume")
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    print("Opening window...")
    plt.show()

# --- 5. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Activity Detector")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--roi", type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), 
                        help="Region of Interest (x y width height)", default=None)
    parser.add_argument("--stride", type=int, default=2, 
                        help="Frame skip stride. Default: 2")
    parser.add_argument("--diagram", action="store_true", 
                        help="If set, displays the Matplotlib diagram.")
    parser.add_argument("--cow-threshold", type=int, default=150000, 
                        help="Pixel count threshold. Default: 150000")
    
    args = parser.parse_args()

    MODE = "color"
    BRIGHT_THRESH = 200
    SAT_THRESH = 80
    VAL_THRESH = 80
    
    roi_obj = None
    if args.roi:
        roi_obj = ROI(*args.roi)

    if os.path.exists(args.video_path):
        cap = cv2.VideoCapture(args.video_path)
        
        try:
            activity_data = compute_activity(
                cap, 
                roi_obj, 
                MODE, 
                BRIGHT_THRESH, 
                SAT_THRESH, 
                VAL_THRESH,
                args.stride,
                args.cow_threshold
            )
            
            cap.release()
            
            if args.diagram and len(activity_data) > 0:
                plot_results(activity_data, MODE, os.path.basename(args.video_path), args.cow_threshold)
            elif len(activity_data) == 0:
                print("No activity detected with current thresholds.")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: File '{args.video_path}' not found.")
