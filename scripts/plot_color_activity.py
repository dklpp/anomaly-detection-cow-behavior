import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
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
        region = frame[roi.slice] if roi else frame
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot colored/bright pixel activity over time for a video ROI."
    )
    parser.add_argument("--video", default="data/cow_trimmed.mp4", help="Path to the input MP4 video.")
    parser.add_argument(
        "--roi",
        type=str,
        default="248,260,420,344",
        help="ROI as x,y,w,h. Use 'full' to analyze entire frame.",
    )
    parser.add_argument(
        "--mode",
        choices=["color", "green"],
        default="color",
        help="Detection mode: 'color' counts saturated colored pixels, 'green' counts bright green pixels.",
    )
    parser.add_argument(
        "--bright-threshold",
        type=int,
        default=100,
        help="Green-channel value (0-255) to consider a pixel part of a cow (green mode).",
    )
    parser.add_argument(
        "--color-sat-threshold",
        type=int,
        default=28,
        help="HSV saturation threshold (0-255) for classifying a pixel as colored (color mode).",
    )
    parser.add_argument(
        "--color-value-threshold",
        type=int,
        default=40,
        help="HSV value (brightness) threshold (0-255) to ignore dim pixels in color mode.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Size of moving-average window (frames) applied to the activity curve.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/color_activity.png",
        help="Path to save the resulting plot PNG.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI for the saved PNG.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving it.",
    )
    args = parser.parse_args()

    if args.roi.lower() == "full":
        roi: Optional[ROI] = None
    else:
        roi = parse_roi(args.roi)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0

    activity = compute_activity(
        cap,
        roi,
        args.mode,
        args.bright_threshold,
        args.color_sat_threshold,
        args.color_value_threshold,
    )
    cap.release()

    if activity.size == 0:
        raise SystemExit("No frames read from the video.")

    frames = np.arange(activity.size)
    times = frames / fps

    smoothed = moving_average(activity, args.smooth_window)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(times, activity, label="Raw", alpha=0.4)
    plt.plot(times, smoothed, label=f"Smoothed (window={args.smooth_window})", linewidth=2)
    plt.xlabel("Time (s)")
    if roi:
        plt.ylabel(f"Pixel count in ROI ({roi.w}x{roi.h})")
    else:
        plt.ylabel("Pixel count (full frame)")

    if args.title:
        plt.title(args.title)
    else:
        roi_label = "full frame" if roi is None else f"ROI x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}"
        plt.title(f"{args.mode.capitalize()} pixel activity\n{os.path.basename(args.video)} — {roi_label}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved plot to: {os.path.abspath(args.output)}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
