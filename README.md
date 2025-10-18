# anomaly-detection-cow-behavior

## Count Cows with Fixed Thresholds

Use `scripts/count_cows.py` to track cow arrivals and departures based on the smoothed pixel count inside the milking-area ROI.

```bash
python scripts/count_cows.py --video data/cow_trimmed.mp4 --dump-signal
```

The script counts the number of “active” pixels in each frame, smooths the curve, and logs:
- Arrival when the smoothed count rises above `--arrive-threshold` (default `20000`).
- Departure when it drops below `--depart-threshold` (default `10000`).

Key options:
- `--arrive-threshold` / `--depart-threshold` set the fixed pixel-count triggers.
- `--smooth-window` controls the moving-average window size.
- `--bright-threshold`, `--color-sat-threshold`, `--color-value-threshold`, and `--roi` match the plotting tool, ensuring consistent preprocessing.

Add `--dump-signal` to print signal stats before trusting the numbers.

### Visualize Activity

Generate a quick plot of colored-pixel intensity over the video with:

```bash
python scripts/plot_color_activity.py --video data/cow_trimmed.mp4 --output output/color_activity.png
```
