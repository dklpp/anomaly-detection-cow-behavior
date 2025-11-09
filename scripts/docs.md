# Script Documentation

## pruning.py
The `pruning.py` script implements a two-stage video filtering system that processes long video recordings of cow behavior at milking stations. It uses OpenCV's MOG2 background subtraction algorithm to identify significant motion events within a specified Region of Interest (ROI). The script analyzes frame-by-frame motion, detecting when cows enter or exit the milking area, and automatically exports these relevant segments to a new video file. It includes configurable parameters for motion thresholds, confirmation frames, and padding time to ensure complete capture of important events while significantly reducing the total video duration for subsequent detailed analysis.
### TODO:
- [] the script now considers any type of motion, we should limit it to consider only the motion of the pixels that represent thermal readings (color). 

## roi_calculator.py
The `roi_calculator.py` script provides an interactive tool for defining the Region of Interest (ROI) in video analysis. When executed, it opens the first frame of a specified video file and allows users to visually select a rectangular area by clicking and dragging the mouse. After selection, it outputs the coordinates (x, y) and dimensions (width, height) of the ROI, which can then be used as configuration parameters in the main pruning script.

## process_video.py
The `process_video.py` script reads the data mp4 file (for testing you can specify argparse to read only N first seconds), then it prunes each frame and saves to the output folder.

## count_cows.py
The `count_cows.py` script analyzes a video to detect cow arrivals and departures within a specified Region of Interest (ROI). It operates by counting pixels that meet certain color criteria ('color' mode using HSV thresholds or 'green' mode for bright green pixels). The script applies a moving average to the pixel count signal to smooth out noise and then uses configurable thresholds to identify 'arrive' and 'depart' events. Finally, it prints a timeline of these events.

## leg_grouping.py
The `leg_grouping.py` script is a pipeline for identifying and categorizing cow legs from a video. It first extracts potential leg images from video frames based on contour properties (size, aspect ratio). Then, it extracts features from these images and uses K-Means clustering to group similar leg images together. The resulting grouped images are saved into separate directories for further analysis.

## plot_color_activity.py
The `plot_color_activity.py` script generates a plot of pixel activity over time for a given video file and ROI. It can track either colored pixels (using HSV thresholds) or bright green pixels. The script calculates the number of target pixels in each frame, applies a moving average to smooth the data, and then generates and saves a plot showing both the raw and smoothed activity curves over time.

## pruning_hsv.py
The `pruning_hsv.py` script is an enhanced version of `pruning.py`. It filters a video to find segments with significant motion, but with an added condition: it only considers motion of pixels that fall within a specific HSV color range, targeting thermal camera colors. It uses background subtraction to detect motion and then filters those motion events with the color mask. The script then exports the relevant, color-filtered motion segments into a new, shorter video file.

## pixel/pixel_counter.py
The `pixel/pixel_counter.py` script is a utility for analyzing a single image. It counts the number of pixels that fall within a predefined HSV color range (designed to identify thermal colors like greens and blues). As output, it prints the total count and saves a new image where the counted pixels are highlighted in red for visual verification.

## pixel/video_pixel_counter.py
The `pixel/video_pixel_counter.py` script analyzes a full video to provide statistics on colored pixel counts per frame. It processes the video frame by frame, counts the pixels within the target HSV color range in each frame, and then calculates and displays aggregate statistics for the entire video, including the maximum, minimum, average, median, and standard deviation of the pixel counts.

## vid-to-image/process_thermal_video.py
The `vid-to-image/process_thermal_video.py` script is designed to detect and capture key moments in a thermal video. It monitors the video for frames where the number of "thermal" pixels exceeds a set threshold. When triggered, it records a 10-second sequence of frames. From this sequence, it generates two summary images: a "fusion" image (the average of all frames) and a collage. The script logs the trigger time and implements a cooldown period to prevent capturing redundant events.

## temperature_anomaly_detection.py
The `temperature_anomaly_detection.py` script analyzes collages of thermal video frames to find temperature-based anomalies. It works by splitting each collage into individual frames and calculating the average and maximum "temperature" (using the V-channel from the HSV colorspace) for the entire collage (global) and for each frame (local). A frame is flagged as an anomaly if its local temperature metrics significantly deviate from the global ones. The script then generates a new collage of all flagged frames, annotated with their timestamps and metrics.

## kmeans_anomaly_detection.py
The `kmeans_anomaly_detection.py` script uses clustering to find visually anomalous frames in the video collages. It extracts a feature vector from every frame across all collages using the original, full-color pixel data. It then trains a KMeans model on this dataset to group visually similar frames. Frames that are far from their assigned cluster's center (specifically, those in the top percentile of distances) are flagged as anomalies. The script outputs a collage of these anomalous frames, annotated with their timestamps and distance scores.
