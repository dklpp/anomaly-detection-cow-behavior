import cv2
import numpy as np
import sys
import os
import argparse
from typing import Optional, List, Tuple
from collections import deque

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
    output_dir: str = None,
    save_stride: int = 10,
    spike_ratio: float = 1.5,
    debug: bool = False
) -> np.ndarray:
    
    counts: List[int] = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    # State tracking
    cow_in_frame = False
    max_pixels_found = 0
    last_event_time = -1.0 

    # --- SPIKE PROTECTION MEMORY ---
    # Keep track of last 30 valid readings to calculate moving average
    pixel_history = deque(maxlen=30)
    consecutive_spikes = 0  # Counter to prevent infinite rejection of valid state changes
    
    # Create output directory if saving is enabled
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved to: {os.path.abspath(output_dir)}")

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
            
        # Skip frames based on processing stride
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Handle ROI logic
        if roi is not None:
            roi_slice_y, roi_slice_x = roi.slice
            # Safety check for boundaries
            if (roi.y + roi.h > frame.shape[0]) or (roi.x + roi.w > frame.shape[1]):
                region = frame[roi.y:min(roi.y+roi.h, frame.shape[0]), 
                               roi.x:min(roi.x+roi.w, frame.shape[1])]
            else:
                region = frame[roi_slice_y, roi_slice_x]
        else:
            region = frame

        # Processing logic
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

        # --- SPIKE DETECTION LOGIC ---
        is_spike = False
        if len(pixel_history) > 5:
            avg_pixels = sum(pixel_history) / len(pixel_history)
            
            # Logic: If current count is significantly larger than average
            # AND the count is significant (above a low noise floor, e.g., 1000)
            if current_count > 1000 and avg_pixels > 0:
                if current_count > (avg_pixels * spike_ratio):
                    is_spike = True

        # Handle persistent state changes (Prevent "Boy who cried wolf" loop)
        if is_spike:
            consecutive_spikes += 1
            # If the "spike" persists for more than 4 processed frames (approx 8-10 raw frames),
            # we assume it's actually a real object (cow) entering and accept it.
            if consecutive_spikes > 4:
                is_spike = False
                consecutive_spikes = 0
                if debug:
                    sys.stdout.write("\033[K")
                    print(f"\n[DEBUG] Persistent signal detected. Accepting new state.")
        else:
            consecutive_spikes = 0
        
        if is_spike:
            # If it's a transient spike, we skip processing
            if debug:
                print(f"[DEBUG] Spike suppressed (Count: {current_count}, Avg: {int(avg_pixels)})")
            
            # Still update progress bar so user knows it's running
            if frame_idx % 20 == 0 and not debug:
                 sys.stdout.write(f"\rScanning: {frame_idx}/{total_frames} | Status: SPIKE SKIP")
                 sys.stdout.flush()

            frame_idx += 1
            continue 
        
        # Add valid reading to history
        pixel_history.append(current_count)
        counts.append(current_count)
        
        # Track max for debugging
        if current_count > max_pixels_found:
            max_pixels_found = current_count

        # --- COW ENTRY/EXIT LOGIC (With 1s Tolerance) ---
        time_sec = frame_idx / fps if fps > 0 else 0

        # Check Entry
        if not cow_in_frame and current_count > cow_pixel_threshold:
            if (time_sec - last_event_time) > 1.0:
                cow_in_frame = True
                last_event_time = time_sec
                sys.stdout.write("\033[K") 
                print(f"\n[EVENT] Cow ENTERED at Frame {frame_idx} ({time_sec:.2f}s) | Pixels: {current_count}")
        
        # Check Exit
        elif cow_in_frame and current_count < cow_pixel_threshold:
            if (time_sec - last_event_time) > 1.0:
                cow_in_frame = False
                last_event_time = time_sec
                sys.stdout.write("\033[K") 
                print(f"\n[EVENT] Cow EXITED at Frame {frame_idx} ({time_sec:.2f}s) | Pixels: {current_count}")

        # --- SAVE FRAME LOGIC ---
        # Only save if cow is detected AND matches the save stride
        if cow_in_frame and output_dir is not None:
            if frame_idx % save_stride == 0:
                filename = os.path.join(output_dir, f"cow_frame_{frame_idx}.jpg")
                
                if debug:
                    # Draw pixel count on the saved image for debugging
                    save_img = region.copy()
                    text = f"Px: {current_count}"
                    # Red text, top-left corner
                    cv2.putText(save_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imwrite(filename, save_img)
                else:
                    cv2.imwrite(filename, region)

        frame_idx += 1
        
        # UPDATED Progress Indicator
        if debug:
            # Print every frame on a new line to track values
            print(f"[DEBUG] Frame {frame_idx}: {current_count} pixels | Status: {'COW' if cow_in_frame else 'EMPTY'}")
        elif frame_idx % 20 == 0: 
            sys.stdout.write(f"\rScanning: {frame_idx}/{total_frames} | Current: {current_count} | Status: {'COW' if cow_in_frame else 'EMPTY'}")
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
                        help="Frame processing skip stride. Default: 2")
    parser.add_argument("--diagram", action="store_true", 
                        help="If set, displays the Matplotlib diagram.")
    parser.add_argument("--cow-threshold", type=int, default=15000, 
                        help="Pixel count threshold. Default: 15000")
    
    # NEW ARGUMENTS
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save frames where cow is detected.")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save every Nth frame when cow is detected. Default: 10")
    parser.add_argument("--spike-ratio", type=float, default=1.5,
                        help="Moving average multiplier to detect spikes. Default: 1.5 (50%% jump)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode: prints pixel count every frame and draws it on saved images.")
    
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
                args.cow_threshold,
                args.output_dir,   # Pass output dir
                args.save_every,   # Pass save stride
                args.spike_ratio,  # Pass spike ratio
                args.debug         # Pass debug flag
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
