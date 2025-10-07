# Script Documentation

## pruning.py
The `pruning.py` script implements a two-stage video filtering system that processes long video recordings of cow behavior at milking stations. It uses OpenCV's MOG2 background subtraction algorithm to identify significant motion events within a specified Region of Interest (ROI). The script analyzes frame-by-frame motion, detecting when cows enter or exit the milking area, and automatically exports these relevant segments to a new video file. It includes configurable parameters for motion thresholds, confirmation frames, and padding time to ensure complete capture of important events while significantly reducing the total video duration for subsequent detailed analysis.
### TODO:
- [] the script now considers any type of motion, we should limit it to consider only the motion of the pixels that represent thermal readings (color). 

## roi_calculator.py
The `roi_calculator.py` script provides an interactive tool for defining the Region of Interest (ROI) in video analysis. When executed, it opens the first frame of a specified video file and allows users to visually select a rectangular area by clicking and dragging the mouse. After selection, it outputs the coordinates (x, y) and dimensions (width, height) of the ROI, which can then be used as configuration parameters in the main pruning script.

## process_video.py
The `process_video.py` script reads the data mp4 file (for testing you can specify argparse to read only N first seconds), then it prunes each frame and saves to the output folder.
