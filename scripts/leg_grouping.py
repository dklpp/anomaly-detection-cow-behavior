import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from pathlib import Path
from tqdm import tqdm
import shutil

VIDEO_PATH = "data/cow_trimmed.mp4"
EXTRACTED_LEGS_DIR = Path("output/legs")
GROUPED_LEGS_DIR = Path("output/grouped_legs")
N_CLUSTERS = 3

for d in [EXTRACTED_LEGS_DIR, GROUPED_LEGS_DIR]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def extract_legs(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    leg_count = 0
    prev_boxes = []

    with tqdm(total=frame_count) as pbar:
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            roi = frame[int(frame_h * 0.6):frame_h, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                if w < 25 or h < 60 or h > 350 or w > 200:
                    continue

                if x > frame_w * 0.75:
                    continue

                aspect = h / float(w)
                if aspect < 1.2 or aspect > 5.5:
                    continue

                duplicate = False
                for (px, py, pw, ph) in prev_boxes[-10:]:
                    if abs(px - x) < 10 and abs(py - y) < 10 and abs(pw - w) < 10 and abs(ph - h) < 10:
                        duplicate = True
                        break
                if duplicate:
                    continue

                prev_boxes.append((x, y, w, h))

                crop = roi[y:y + h, x:x + w]
                save_path = EXTRACTED_LEGS_DIR / f"leg_{frame_idx}_{x}_{y}.png"
                cv2.imwrite(str(save_path), crop)
                leg_count += 1

            pbar.update(1)

    cap.release()
    print(f"Extracted {leg_count} clean leg candidates.")

def extract_features():
    feats, paths = [], []
    for f in sorted(EXTRACTED_LEGS_DIR.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, (64, 64))
        feat = resized.flatten().astype(np.float32) / 255.0
        feats.append(feat)
        paths.append(f)
    print(f"✅ {len(feats)} features extracted.")
    return np.array(feats), paths

def cluster_and_save(features, paths):
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    print("Legs grouped into clusters:", np.bincount(labels))

    for i in range(N_CLUSTERS):
        (GROUPED_LEGS_DIR / f"group_{i}").mkdir(parents=True, exist_ok=True)

    for path, label in zip(paths, labels):
        new_path = GROUPED_LEGS_DIR / f"group_{label}" / path.name
        path.rename(new_path)

if __name__ == "__main__":
    print("Starting leg grouping pipeline...\n")

    extract_legs(VIDEO_PATH)
    features, paths = extract_features()

    if len(features) == 0:
        print("No legs found.")
    else:
        cluster_and_save(features, paths)

    print("\n Done! Check 'output/grouped_legs/'.")
