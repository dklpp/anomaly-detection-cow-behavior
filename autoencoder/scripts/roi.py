import cv2
import argparse
import os

def play_video(video_path, roi=None, select_mode=False):
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' was not found.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # --- INTERACTIVE SELECTION MODE ---
    if select_mode:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read the first frame to select ROI.")
            return
        
        print("\n--- SELECTION MODE ---")
        print("1. Draw a box with your mouse.")
        print("2. Press SPACE or ENTER to confirm.")
        print("3. Press 'c' to cancel selection.")
        
        # Select ROI returns a tuple (x, y, w, h)
        # It opens a temporary window for selection
        selection = cv2.selectROI("Select ROI (Press Enter to Confirm)", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI (Press Enter to Confirm)")

        # Check if a valid selection was made (w and h > 0)
        if selection[2] > 0 and selection[3] > 0:
            roi = selection
            x, y, w, h = roi
            print("\n" + "="*40)
            print(f"captured ROI Coordinates: {x} {y} {w} {h}")
            print(f"Command to reuse this:\npython script.py {video_path} --roi {x} {y} {w} {h}")
            print("="*40 + "\n")
            
            # Reset video to start for playback
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Selection cancelled or invalid. Playing full video.")

    # --- PLAYBACK LOOP ---
    print(f"Playing: {video_path}")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        display_frame = frame

        # Apply ROI cropping if coordinates exist (either from CLI or Selection)
        if roi:
            x, y, w, h = roi
            
            # Sanity Check bounds
            max_h, max_w, _ = frame.shape
            if x >= max_w or y >= max_h:
                # If ROI is invalid, just show full frame to prevent crash
                pass 
            else:
                display_frame = frame[y:y+h, x:x+w]

        if display_frame.size == 0:
             continue

        cv2.imshow('Video Player', display_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play video with optional ROI selection.")
    
    parser.add_argument("input_video", help="Path to the video file")
    
    # Group to ensure user doesn't try to provide coordinates AND select at the same time
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument(
        "--roi", 
        type=int, 
        nargs=4, 
        metavar=('X', 'Y', 'W', 'H'),
        help="Manual ROI: x y width height"
    )
    
    group.add_argument(
        "--select", 
        action="store_true", 
        help="Interactively select the ROI from the first frame"
    )

    args = parser.parse_args()

    play_video(args.input_video, args.roi, args.select)
