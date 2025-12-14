link to the output videos : https://epitafr-my.sharepoint.com/:f:/g/personal/aniss1_outaleb_epita_fr/IgBQnE8mnTi7TZD0mtWL2MRtAcNmmEnScBp990gGX5Zr_M4?e=SvhqIx
# Individual Report

## Step 1: Initial Setup and Object Detection

In the first step of the project, I focused on setting up the environment and utilizing the provided object detection code. This involved the following tasks:

1. **Environment Setup**: I ensured that the necessary Python environment was configured, and all required libraries such as OpenCV and NumPy were installed.
2. **Understanding the Provided Object Detection Code**: The `detect` function in the `Detector.py` file was provided. This function processes video frames to detect objects using edge detection and contour analysis. The detected objects are filtered based on their size to ensure only valid objects are considered.

## File architecture

Below is the project file structure and the files associated with each part of the practical work:

- **Main project files**:
   - `Detector.py` — implements the `detect()` function (Canny + contour detection). Used to extract object centroids/patches (TP1, TP2, TP3, TP4).
   - `KalmanFilter.py` — contains the `KalmanFilter` class with `__init__()`, `predict()` and `update()` (TP1; reused in TP3 and TP4).
   - `objTracking.py` — main script to run Single Object Tracking experiments (TP1) and provides visualization utilities.
   - `iou.py` — utility functions to compute Intersection over Union (IoU) between bounding boxes (TP2).
   - `iou_Kalman.py` — IoU tracker integrated with Kalman filter (TP3).
   - `iou_Kalman_ReID.py` — IoU+Kalman extension with Re-Identification (TP4).
   - `reid_osnet_x025_market1501.onnx` — ReID model (OSNet) used for appearance feature extraction (TP4).

- **Data and sequences**:
   - `ADL-Rundle-6/` — image sequence (`img1/`), ground truth (`gt/`) and detection files (`det/`) used for TP2, TP3 and TP4.
   - `2D_Kalman-Filter_TP1/video/` — example videos to test TP1 tracking.

## Step 2: Kalman Filter Implementation and Object Tracking

In the second step, I implemented the Kalman Filter to track detected objects over time. This involved the following tasks:

1. **Kalman Filter Implementation**: I developed the `KalmanFilter` class in the `KalmanFilter.py` file. The Kalman Filter predicts the next state of the object (position and velocity) and updates its estimates based on the detected measurements.
2. **Integration with Object Detection**: I integrated the Kalman Filter with the object detection pipeline in the `objTracking.py` file. The detected object positions are used as measurements to update the Kalman Filter's state.
3. **Visualization of Tracking Results**: I visualized the tracking results by drawing the detected, predicted, and estimated positions on the video frames. The trajectory of the tracked object was also displayed.

### Challenges Encountered
- **Challenge**: Ensuring the Kalman Filter parameters (e.g., process noise, measurement noise) were tuned correctly for stable tracking.
- **Solution**: I adjusted the noise covariance matrices and tested the system with different scenarios to achieve optimal tracking performance.

This step demonstrated the effectiveness of the Kalman Filter in maintaining accurate object tracking even with noisy measurements.

## Step 3: IoU-Based Tracking (Bounding-Box Tracker)

In the third step of the project, I developed an IoU-based tracker to perform multiple object tracking (MOT) using bounding boxes. This involved the following tasks:

1. **Loading Detections**: I loaded pre-generated detections from a MOT-challenge formatted text file. Each detection contained information such as frame number, bounding box coordinates, and confidence score. The detections were stored in a dictionary for efficient access during tracking.

2. **Creating the Similarity Matrix**: For each frame, I calculated the Intersection over Union (IoU) between tracked objects and new detections. This was done by creating a similarity matrix, where each entry represented the IoU value between a tracked object and a detection. The matrix dimensions were determined by the number of tracked objects and detections in the current frame.

3. **Associating Detections to Tracks**: Using the Hungarian algorithm (via the `linear_sum_assignment` function from the `scipy` library), I found the optimal assignment of detections to tracked objects based on the similarity matrix. This ensured that each detection was assigned to the most appropriate track.

4. **Track Management**: I implemented logic to manage tracks over time:
   - **Matched Tracks**: Updated existing tracks with the assigned detections.
   - **Unmatched Tracks**: Incremented the missed frame count for tracks without a match. Tracks exceeding the maximum allowed missed frames were removed.
   - **Unmatched Detections**: Created new tracks for detections that were not assigned to any existing track.

5. **Visualization of Tracking Results**: I developed an interface to visualize the tracking results. This included:
   - Displaying video frames with overlaid bounding boxes for each tracked object.
   - Labeling each bounding box with its unique track ID for identification.
   - Saving the tracking results as a video file for further analysis.

6. **Saving Tracking Results**: The tracking results were saved in a text file in the MOT-challenge format. Each line represented one object instance per frame, including the frame number, track ID, bounding box coordinates, and a confidence flag.

### Challenges Encountered
- **Challenge**: Ensuring accurate IoU calculations and handling edge cases where no detections or tracks were present.
- **Solution**: I added validations to handle empty similarity matrices and ensured that unmatched tracks and detections were managed correctly.

- **Challenge**: Visualizing and debugging the tracking process to ensure the tracker performed as expected.
- **Solution**: I added debugging print statements and visual overlays to verify the correctness of the tracking logic.

This step demonstrated the effectiveness of IoU-based tracking for multiple object tracking and provided insights into the challenges of real-world tracking scenarios.

## Step 4: Integration of Kalman Filter with IoU-Based Tracking

In the fourth step of the project, I integrated the Kalman Filter with the IoU-based tracking system to enhance the tracking accuracy and robustness. This involved the following tasks:

1. **Integration of Kalman Filter**: Each track was equipped with a Kalman Filter to predict its next state (position and velocity) based on its previous state and measurements. The Kalman Filter was initialized with appropriate parameters for process noise and measurement noise.

2. **Representation of Bounding Boxes as Centroids**: To ensure compatibility with the Kalman Filter, bounding boxes were represented by their centroids. This allowed the Kalman Filter to predict and update the position of each track effectively.

3. **Prediction Step**: Before associating detections to tracks, the Kalman Filter predicted the next state of each track. This prediction was used to calculate the similarity matrix.

4. **Updated IoU Calculation**: The IoU calculation was modified to use the predicted states of the tracks. Instead of using bounding box coordinates directly, the IoU was calculated based on the Euclidean distance between the centroids of tracks and detections.

5. **Track Management**: The track management logic was updated to incorporate the Kalman Filter:
   - **Matched Tracks**: Updated the Kalman Filter state with the assigned detections.
   - **Unmatched Tracks**: Used the Kalman Filter's prediction to maintain the track's state even when no detection was assigned.
   - **Unmatched Detections**: Created new tracks with initialized Kalman Filters for detections that were not assigned to any existing track.

6. **Visualization and Saving Results**: The visualization interface was updated to display the predicted and updated positions of tracks. The tracking results were saved in the MOT-challenge format, including the predicted positions.

### Challenges Encountered
- **Challenge**: Ensuring the Kalman Filter parameters were tuned correctly for stable tracking.
- **Solution**: I adjusted the noise covariance matrices and tested the system with different scenarios to achieve optimal tracking performance.

- **Challenge**: Modifying the IoU calculation to use predicted states while maintaining compatibility with the Hungarian algorithm.
- **Solution**: I represented bounding boxes as centroids and calculated similarity based on Euclidean distances, which were converted to IoU-like scores.

This step demonstrated the effectiveness of combining Kalman Filter predictions with IoU-based tracking for robust multiple object tracking.

## Step 5: Integration of Object Re-Identification (ReID) with IoU-Kalman Tracker

In the fifth step of the project, I extended the IoU-Kalman tracker to include object re-identification (ReID) for more robust tracking. This involved the following tasks:

1. **Feature Extraction**: I implemented a feature extraction pipeline using a pre-trained lightweight deep learning model (OSNet). The model extracts appearance features from detected objects (image patches) to enable ReID. The patches were resized to (64, 128) to match the input size used during the model's training.

2. **Patch Preprocessing**: Each detected object was preprocessed before feature extraction:
   - **Patch Generation**: Generated patches for each detected object based on its bounding box.
   - **Patch Resizing**: Resized patches to (64, 128) to match the ReID model's input size.
   - **Color Conversion**: Converted patches from BGR to RGB format.
   - **Normalization**: Normalized patches using the mean and standard deviation values of the Market1501 dataset.

3. **ReID Feature Computation**: For each patch, the ReID model computed a feature vector representing the object's appearance. These feature vectors were normalized to ensure consistent comparison.

4. **Similarity Metrics for ReID**: I used cosine similarity to compare feature vectors of detected objects with those of tracked objects. This provided a measure of appearance similarity between objects.

5. **Combining IoU and Feature Similarity**: To make the association process more robust, I combined geometric similarity (IoU) and appearance similarity (ReID) using a weighted sum:
   \[
   S = \alpha \cdot IoU + \beta \cdot Normalized\ Similarity
   \]
   where:
   - \( \alpha \) and \( \beta \) are tunable weights.
   - Normalized Similarity is computed as:
     \[
     Normalized\ Similarity = \frac{1}{1 + Euclidean\ Distance}
     \]
     This ensures that a lower distance results in a higher similarity score.

6. **Track Association**: The combined similarity score was used to associate detections with tracks using the Hungarian algorithm. This improved the tracker’s ability to handle occlusions and re-identify objects across frames.

### Challenges Encountered
- **Challenge**: Ensuring the ReID model was correctly integrated and optimized for feature extraction.
- **Solution**: I tested the model with various scenarios and adjusted preprocessing steps to ensure compatibility with the ReID model.

- **Challenge**: Balancing the weights \( \alpha \) and \( \beta \) for optimal association performance.
- **Solution**: I experimented with different weight values and evaluated the tracker’s performance to find the best configuration.

This step demonstrated the effectiveness of combining appearance and geometric information for robust multiple object tracking.