import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    @property
    def slice(self) -> Tuple[slice, slice]:
        return (slice(self.y, self.y + self.h), slice(self.x, self.x + self.w))


def parse_roi(value: Optional[str]) -> Optional[ROI]:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must have four comma-separated integers (x,y,w,h)")
    try:
        x, y, w, h = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return ROI(x, y, w, h)


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(signal, (window // 2,), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: signal.shape[0]]


def compute_activity(
    cap: cv2.VideoCapture,
    roi: Optional[ROI],
    mode: str,
    bright_threshold: int,
    color_sat_threshold: int,
    color_value_threshold: int,
) -> np.ndarray:
    counts: List[int] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if roi is not None:
            roi_slice_y, roi_slice_x = roi.slice
            region = frame[roi_slice_y, roi_slice_x]
        else:
            region = frame

        if mode == "green":
            green = region[:, :, 1]
            counts.append(int(np.count_nonzero(green >= bright_threshold)))
        elif mode == "color":
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            mask = (sat >= color_sat_threshold) & (val >= color_value_threshold)
            counts.append(int(np.count_nonzero(mask)))
        else:
            raise ValueError(f"Unsupported detection mode: {mode}")
    return np.asarray(counts, dtype=np.float32)


def detect_events(
    signal: np.ndarray,
    fps: float,
    enter_threshold: float,
    exit_threshold: float,
) -> Tuple[List[Tuple[str, int, float]], List[Tuple[str, int, float]]]:
    events: List[Tuple[str, int, float]] = []
    present = False
    for frame_idx, value in enumerate(signal):
        if not present and value >= enter_threshold:
            present = True
            events.append(("arrive", frame_idx, frame_idx / fps))
        elif present and value <= exit_threshold:
            present = False
            events.append(("depart", frame_idx, frame_idx / fps))
    arrivals = [e for e in events if e[0] == "arrive"]
    departures = [e for e in events if e[0] == "depart"]
    return arrivals, departures


def save_event_frames(
    video_path: str,
    roi: Optional[ROI],
    arrivals: List[Tuple[str, int, float]],
    departures: List[Tuple[str, int, float]],
    output_root: str = "output",
) -> None:
    os.makedirs(output_root, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Pair arrival→departure events
    pairs = []
    for a, d in zip(arrivals, departures):
        _, a_idx, _ = a
        _, d_idx, _ = d
        if d_idx > a_idx:
            pairs.append((a_idx, d_idx))

    if not pairs:
        print("[INFO] No arrival/departure pairs detected — nothing to extract.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Failed to reopen video for extraction.")
        return

    for event_i, (start_f, end_f) in enumerate(pairs, start=1):
        cow_dir = os.path.join(output_dir, f"cow_{event_i:02d}")
        os.makedirs(cow_dir, exist_ok=True)

        print(f"[INFO] Saving event {event_i}: frames {start_f} → {end_f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_count = 0

        while True:
            cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur_frame > end_f:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if roi:
                sy, sx = roi.slice
                frame = frame[sy, sx]

            if frame.size == 0:
                continue

            output_path = os.path.join(cow_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1

    cap.release()
    print(f"[DONE] Frames saved inside: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count cow arrivals/departures and export cropped frames between events."
    )
    parser.add_argument("--video", default="data/cow_hsv_pruned_1h.mp4", help="Path to the input MP4 video.")
    parser.add_argument(
        "--roi",
        type=str,
        #default="248,260,420,344",
        default="253,256,418,343",
        help="ROI as x,y,w,h. Use 'full' to analyze entire frame.",
    )
    parser.add_argument(
        "--mode",
        choices=["color", "green"],
        default="color",
        help="Detection mode: 'color' counts saturated colored pixels, 'green' counts bright green pixels.",
    )
    parser.add_argument("--bright-threshold", type=int, default=100)
    parser.add_argument("--color-sat-threshold", type=int, default=28)
    parser.add_argument("--color-value-threshold", type=int, default=40)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--arrive-threshold", type=float, default=20000.0)
    parser.add_argument("--depart-threshold", type=float, default=10000.0)
    parser.add_argument("--dump-signal", action="store_true")
    args = parser.parse_args()

    if args.roi.lower() == "full":
        roi = None
    else:
        roi = parse_roi(args.roi)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    raw_signal = compute_activity(
        cap,
        roi,
        args.mode,
        args.bright_threshold,
        args.color_sat_threshold,
        args.color_value_threshold,
    )
    cap.release()

    if raw_signal.size == 0:
        raise SystemExit("No frames read from the video.")

    smoothed_signal = moving_average(raw_signal, args.smooth_window)

    if args.dump_signal:
        print(f"Frames: {smoothed_signal.size}")
        print(f"Min/Max/Mean: {smoothed_signal.min():.0f} / {smoothed_signal.max():.0f} / {smoothed_signal.mean():.0f}")

    arrivals, departures = detect_events(
        smoothed_signal,
        fps,
        args.arrive_threshold,
        args.depart_threshold,
    )

    print("\n=== Cow Traffic Report ===")
    print(f"Arrivals:   {len(arrivals)}")
    print(f"Departures: {len(departures)}")

    # ✅ Extract training frames
    save_event_frames(
        video_path=args.video,
        roi=roi,
        arrivals=arrivals,
        departures=departures,
        output_root="output",
    )


if __name__ == "__main__":
    main()
