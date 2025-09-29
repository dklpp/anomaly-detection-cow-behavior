import cv2

def get_roi_interactively(video_path):
    """
    Opens the first frame of the video and allows the user to select the ROI
    by clicking and dragging a rectangle.
    
    Instructions:
    1. A window will appear showing the first frame.
    2. Click and drag your mouse to draw a box around the milking station entrance.
    3. Press ENTER or SPACE to confirm the selection.
    4. Press 'c' to cancel the selection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return None

    print("\n--- ROI SELECTION ACTIVE ---")
    print("Draw a box around the cow entrance, then press ENTER or SPACE.")
    
    # SelectROI returns (x, y, w, h)
    roi = cv2.selectROI("ROI Selection Tool (Press ENTER/SPACE to Confirm)", frame, False, False)
    
    cv2.destroyAllWindows()
    cap.release()
    
    if roi and roi[2] > 0 and roi[3] > 0:
        x, y, w, h = [int(i) for i in roi]
        print("\n========================================================")
        print("✅ ROI Found! Copy these values back into the CONFIG section:")
        print(f"ROI_X: {x}, ROI_Y: {y}, ROI_W: {w}, ROI_H: {h}")
        print("========================================================\n")
        return x, y, w, h
    else:
        print("ROI selection cancelled or failed.")
        return None

get_roi_interactively("cow_trimmed.mp4")