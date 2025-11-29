import cv2
import numpy as np
import os
import re
from pathlib import Path
import math
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import datetime

# --- Configuration ---
COLLAGE_DIR = Path("output_sequences")
TIMESTAMPS_FILE = COLLAGE_DIR / "capture_timestamps.txt"
OUTPUT_DIR = Path("output/kmeans_anomalies")
ANOMALY_COLLAGE_PATH = OUTPUT_DIR / "kmeans_anomaly_collage.png"

N_CLUSTERS = 8
IMG_RESIZE_DIM = (1024, 1024)
FRAME_RESIZE_DIM = (256, 256)

# Anomaly is defined as being in the top X percentile of distances to cluster center
ANOMALY_PERCENTILE_THRESHOLD = 97

ANOMALIES_FILE = Path("kmeans/anomalies.txt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_timestamp_for_frame(trigger_frame_num):
    """Parses the timestamps file to find the time for a given trigger frame."""
    if not TIMESTAMPS_FILE.exists():
        return "Timestamp file not found"
    
    with open(TIMESTAMPS_FILE, 'r') as f:
        for line in f:
            match = re.search(r"video time: (\S+) \(Frame: (\d+)\)", line)
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
    frame_h, frame_w = IMG_RESIZE_DIM
    
    grid_rows = math.ceil(num_frames / grid_cols)
    collage_h = grid_rows * frame_h
    collage_w = grid_cols * frame_w
    
    final_collage = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)

    for i, data in enumerate(flagged_frames_data):
        row = i // grid_cols
        col = i % grid_cols
        
        y_offset = row * frame_h
        x_offset = col * frame_w
        
        # Resize frame before placing it in the collage
        resized_frame = cv2.resize(data['frame'], (frame_w, frame_h))
        final_collage[y_offset:y_offset+frame_h, x_offset:x_offset+frame_w] = resized_frame

    cv2.imwrite(str(output_path), final_collage)
    print(f"Anomaly collage saved to: {output_path}")

def main():
    collage_files = sorted(list(COLLAGE_DIR.glob("collage_triggered_at_frame_*.png")))
    print(f"Found {len(collage_files)} collage files.")

    # Load ground truth anomalies
    known_anomalies = set()
    if ANOMALIES_FILE.exists():
        with open(ANOMALIES_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    known_anomalies.add((parts[0], int(parts[1])))
    print(f"Loaded {len(known_anomalies)} known anomalies for evaluation.")

    all_frames_features = []
    all_frames_metadata = []
    y_true = []

    print("Step 1: Extracting features from all frames...")
    for collage_path in tqdm(collage_files):
        trigger_frame_match = re.search(r"frame_(\d+)\.png", collage_path.name)
        if not trigger_frame_match:
            continue
        trigger_frame = int(trigger_frame_match.group(1))
        
        collage_img = cv2.imread(str(collage_path))
        if collage_img is None:
            continue

        frames = split_collage(collage_img)
        timestamp = get_timestamp_for_frame(trigger_frame)

        for i, frame in enumerate(frames):
            # Resize frame to a fixed size to ensure consistent feature vector length
            resized_frame = cv2.resize(frame, FRAME_RESIZE_DIM)
            # Use the original BGR frame, not resized or grayscale
            feature_vector = resized_frame.flatten().astype(np.float32) / 255.0
            
            all_frames_features.append(feature_vector)
            all_frames_metadata.append({
                'original_frame': frame,
                'collage_source': collage_path.name,
                'frame_index': i,
                'timestamp': timestamp
            })
            
            is_anomaly = 1 if (collage_path.name, i) in known_anomalies else 0
            y_true.append(is_anomaly)
    
    if not all_frames_features:
        print("No features were extracted. Exiting.")
        return

    y_true = np.array(y_true)
    features_array = np.array(all_frames_features)
    print(f"\nStep 2: Running KMeans with {N_CLUSTERS} clusters on {len(features_array)} frames...")
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(features_array)
    print("KMeans training complete.")

    print("\nStep 3: Calculating distances and finding anomalies...")
    # Get the cluster center for each frame
    transformed_features = kmeans.transform(features_array)
    distances = np.min(transformed_features, axis=1)

    # Find the distance threshold for anomalies
    distance_threshold = np.percentile(distances, ANOMALY_PERCENTILE_THRESHOLD)
    print(f"Anomaly distance threshold ({ANOMALY_PERCENTILE_THRESHOLD}th percentile): {distance_threshold:.4f}")

    # Create prediction array
    y_pred = (distances > distance_threshold).astype(int)

    # Filter for anomalies
    anomaly_indices = np.where(y_pred == 1)[0]
    print(f"Found {len(anomaly_indices)} potential anomalies.")

    # --- Evaluation ---
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    print("\n--- Evaluation Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("------------------------\n")

    anomalous_frames_for_collage = []
    markdown_report_lines = [
        "# KMeans Anomaly Detection Report",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Anomaly Threshold: {ANOMALY_PERCENTILE_THRESHOLD}th percentile",
        f"Found {len(anomaly_indices)} potential anomalies.",
        "",
        "## Evaluation Metrics",
        f"- **Precision:** {precision:.4f}",
        f"- **Recall:** {recall:.4f}",
        f"- **F1-Score:** {f1:.4f}",
        "",
        "## Detected Anomalies",
        "| Source Collage | Frame Index | Timestamp | Score |",
        "|---|---|---|---|"
    ]

    for idx in anomaly_indices:
        metadata = all_frames_metadata[idx]
        distance_score = distances[idx]
        
        report_line = f"| {metadata['collage_source']} | {metadata['frame_index']} | {metadata['timestamp']} | {distance_score:.4f} |"
        markdown_report_lines.append(report_line)

        info_frame = metadata['original_frame'].copy()
        text_ts = f"Time: {metadata['timestamp']}"
        text_metrics = f"Score: {distance_score:.4f}"
        
        cv2.rectangle(info_frame, (0, 0), (200, 40), (0, 0, 0), -1)
        
        cv2.putText(info_frame, text_ts, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(info_frame, text_metrics, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        anomalous_frames_for_collage.append({
            'frame': info_frame
        })

    # --- Final Collage Creation ---
    print("\nStep 4: Creating final anomaly collage...")
    create_results_collage(anomalous_frames_for_collage, ANOMALY_COLLAGE_PATH)

    print("Step 5: Writing markdown report...")
    report_path = OUTPUT_DIR / "anomalies_report.md"
    markdown_report_lines.append("\n## Anomaly Collage")
    markdown_report_lines.append(f"![Anomaly Collage]({ANOMALY_COLLAGE_PATH.name})")

    with open(report_path, 'w') as f:
        f.write("\n".join(markdown_report_lines))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
