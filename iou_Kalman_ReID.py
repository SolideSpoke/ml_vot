import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import cv2
import glob
from KalmanFilter import KalmanFilter
import torch
import torchvision.transforms as transforms
from PIL import Image


db = "public-dataset"


def load_mot_detections(file_path):
    detections = {}
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            if len(fields) != 10:
                continue
            frame = int(fields[0])
            bb_left = float(fields[2])
            bb_top = float(fields[3])
            bb_width = float(fields[4])
            bb_height = float(fields[5])
            centroid_x = bb_left + bb_width / 2
            centroid_y = bb_top + bb_height / 2
            detection = {
                'frame': frame,
                'id': int(fields[1]),
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'conf': float(fields[6]),
                'x': float(fields[7]),
                'y': float(fields[8]),
                'z': float(fields[9])
            }
            if frame not in detections:
                detections[frame] = []
            detections[frame].append(detection)
    return detections


class ReIDFeatureExtractor:
    def __init__(self, model_path=None, roi_size=(128, 64)):
        self.roi_height, self.roi_width = roi_size
        self.roi_means = np.array([0.485, 0.456, 0.406])
        self.roi_stds = np.array([0.229, 0.224, 0.225])
        try:
            if model_path and os.path.exists(model_path):
                self.model = torch.load(model_path)
            else:
                from torchvision import models
                self.model = models.resnet18(pretrained=True)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            print(f"ReID model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load ReID model: {e}")
            self.model = None
    def preprocess_patch(self, im_crop):
        roi_input = cv2.resize(im_crop, (self.roi_width, self.roi_height))
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        roi_input = np.asarray(roi_input).astype(np.float32) / 255.0
        roi_input = (roi_input - self.roi_means) / self.roi_stds
        roi_input = np.moveaxis(roi_input, -1, 0)
        object_patch = torch.from_numpy(roi_input.astype(np.float32))
        return object_patch
    def extract_features(self, image, bbox):
        if self.model is None:
            return np.random.randn(512)
        x1 = max(0, int(bbox['bb_left']))
        y1 = max(0, int(bbox['bb_top']))
        x2 = min(image.shape[1], int(bbox['bb_left'] + bbox['bb_width']))
        y2 = min(image.shape[0], int(bbox['bb_top'] + bbox['bb_height']))
        if x2 <= x1 or y2 <= y1:
            return np.random.randn(512)
        im_crop = image[y1:y2, x1:x2]
        patch_tensor = self.preprocess_patch(im_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(patch_tensor)
            features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-6)
        return features


def calculate_iou(boxA, boxB):
    xA1, yA1 = boxA['bb_left'], boxA['bb_top']
    xA2, yA2 = xA1 + boxA['bb_width'], yA1 + boxA['bb_height']
    xB1, yB1 = boxB['bb_left'], boxB['bb_top']
    xB2, yB2 = xB1 + boxB['bb_width'], yB1 + boxB['bb_height']
    xI1, yI1 = max(xA1, xB1), max(yA1, yB1)
    xI2, yI2 = min(xA2, xB2), min(yA2, yB2)
    interWidth = max(0, xI2 - xI1)
    interHeight = max(0, yI2 - yI1)
    interArea = interWidth * interHeight
    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def cosine_similarity(feat1, feat2):
    dot_product = np.dot(feat1, feat2)
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cosine_sim = dot_product / (norm1 * norm2)
    normalized_sim = (cosine_sim + 1) / 2
    return normalized_sim


def create_similarity_matrix(tracked_boxes, detected_boxes, frame, reid_extractor, 
                            alpha_geometric=0.6, alpha_appearance=0.4):
    if not tracked_boxes or not detected_boxes:
        return np.empty((len(tracked_boxes), len(detected_boxes)))
    for track in tracked_boxes:
        track.predict()
    detection_features = []
    for det in detected_boxes:
        features = reid_extractor.extract_features(frame, det)
        detection_features.append(features)
        det['features'] = features
    t_centroids = np.array([[t.bbox['centroid_x'], t.bbox['centroid_y']] for t in tracked_boxes])
    d_centroids = np.array([[d['centroid_x'], d['centroid_y']] for d in detected_boxes])
    distances = np.linalg.norm(t_centroids[:, None] - d_centroids, axis=2)
    max_distance = 100.0
    geometric_similarity = np.maximum(0, 1 - distances / max_distance)
    appearance_similarity = np.zeros((len(tracked_boxes), len(detected_boxes)))
    for i, track in enumerate(tracked_boxes):
        if track.features is not None:
            for j, det_feat in enumerate(detection_features):
                appearance_similarity[i, j] = cosine_similarity(track.features, det_feat)
    combined_similarity = (alpha_geometric * geometric_similarity + 
                          alpha_appearance * appearance_similarity)
    return combined_similarity


def associate_detections_to_tracks(similarity_matrix, iou_threshold=0.3):
    num_tracks = similarity_matrix.shape[0]
    num_detections = similarity_matrix.shape[1]
    if num_tracks == 0:
        return [], [], list(range(num_detections))
    if num_detections == 0:
        return [], list(range(num_tracks)), []
    cost_matrix = 1 - similarity_matrix
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    matched_indices = []
    unmatched_tracks = list(range(num_tracks))
    unmatched_detections = list(range(num_detections))
    for t, d in zip(track_indices, detection_indices):
        if similarity_matrix[t, d] >= iou_threshold:
            matched_indices.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)
    return matched_indices, unmatched_tracks, unmatched_detections


class Track:
    def __init__(self, track_id, bbox, max_missed_frames=5):
        self.track_id = track_id
        self.bbox = bbox.copy()
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames
        self.features = bbox.get('features', None)
        self.kalman_filter = KalmanFilter(
            dt=1, 
            u_x=0, 
            u_y=0, 
            std_acc=1, 
            x_std_meas=0.1, 
            y_std_meas=0.1
        )
        self.kalman_filter.x = np.array([
            [bbox['centroid_x']],
            [bbox['centroid_y']],
            [0],
            [0]
        ])
    def predict(self):
        predicted_state = self.kalman_filter.predict()
        self.bbox['centroid_x'] = predicted_state[0, 0]
        self.bbox['centroid_y'] = predicted_state[1, 0]
    def update(self, bbox):
        self.bbox = bbox.copy()
        if 'features' in bbox and bbox['features'] is not None:
            if self.features is None:
                self.features = bbox['features']
            else:
                self.features = 0.8 * self.features + 0.2 * bbox['features']
                self.features = self.features / (np.linalg.norm(self.features) + 1e-6)
        measurement = np.array([[bbox['centroid_x']], [bbox['centroid_y']]])
        updated_state = self.kalman_filter.update(measurement)
        self.bbox['centroid_x'] = updated_state[0, 0]
        self.bbox['centroid_y'] = updated_state[1, 0]
        self.missed_frames = 0
    def increment_missed(self):
        self.missed_frames += 1
    def is_active(self):
        return self.missed_frames <= self.max_missed_frames


class Tracker:
    def __init__(self, max_missed_frames=5, reid_extractor=None):
        self.tracks = []
        self.next_track_id = 1
        self.max_missed_frames = max_missed_frames
        self.reid_extractor = reid_extractor
    def manage_tracks(self, matched, unmatched_tracks, unmatched_detections, detected_boxes):
        for track_idx, detection_idx in matched:
            if track_idx < len(self.tracks) and detection_idx < len(detected_boxes):
                self.tracks[track_idx].update(detected_boxes[detection_idx])
        for track_idx in unmatched_tracks:
            if track_idx < len(self.tracks):
                self.tracks[track_idx].increment_missed()
        self.tracks = [track for track in self.tracks if track.is_active()]
        for detection_idx in unmatched_detections:
            if detection_idx < len(detected_boxes):
                detection = detected_boxes[detection_idx]
                new_track = Track(
                    track_id=self.next_track_id, 
                    bbox=detection,
                    max_missed_frames=self.max_missed_frames
                )
                self.tracks.append(new_track)
                self.next_track_id += 1


def save_tracking_results(all_frame_tracks, sequence_name, output_dir):
    output_path = f"{output_dir}/{sequence_name}.txt"
    with open(output_path, 'w') as file:
        for frame_id in sorted(all_frame_tracks.keys()):
            tracks = all_frame_tracks[frame_id]
            for track in tracks:
                bbox = track.bbox
                track_id = track.track_id
                x, y, z = -1, -1, -1
                file.write(
                    f"{frame_id},{track_id},{bbox['bb_left']:.2f},{bbox['bb_top']:.2f},"
                    f"{bbox['bb_width']:.2f},{bbox['bb_height']:.2f},1,{x},{y},{z}\n"
                )


if __name__ == "__main__":
    base_folder = "ADL-Rundle-6"
    folder = os.path.join(base_folder, "det")
    img_folder = os.path.join(base_folder, "img1")
    bbox_file = os.path.join(folder, "public-dataset", "det.txt")
    print(f"Bounding box file exists: {os.path.exists(bbox_file)}")
    output_video_path = os.path.join(folder, "output_video_reid.avi")
    output_txt_path = os.path.join(folder, "public-dataset")
    print("Initializing ReID feature extractor...")
    reid_extractor = ReIDFeatureExtractor(roi_size=(128, 64))
    tracker = Tracker(max_missed_frames=5, reid_extractor=reid_extractor)
    detections = load_mot_detections(bbox_file)
    print("Detections loaded:", len(detections), "frames")
    image_files = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
    print("Image files found:", len(image_files))
    all_frame_tracks = {}
    if image_files:
        first_image = cv2.imread(image_files[0])
        frame_height, frame_width, _ = first_image.shape
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        for frame_idx, image_file in enumerate(image_files, start=1):
            frame = cv2.imread(image_file)
            detected_boxes = detections.get(frame_idx, [])
            if frame_idx % 50 == 0:
                print(f"Processing frame {frame_idx}/{len(image_files)}")
            similarity_matrix = create_similarity_matrix(
                tracker.tracks, 
                detected_boxes, 
                frame,
                reid_extractor,
                alpha_geometric=0.6,
                alpha_appearance=0.4
            )
            matched, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(
                similarity_matrix, iou_threshold=0.3
            )
            tracker.manage_tracks(matched, unmatched_tracks, unmatched_detections, detected_boxes)
            all_frame_tracks[frame_idx] = []
            for track in tracker.tracks:
                if track.is_active():
                    track_copy = Track(track.track_id, track.bbox.copy(), track.max_missed_frames)
                    track_copy.features = track.features
                    all_frame_tracks[frame_idx].append(track_copy)
            for track in tracker.tracks:
                if track.is_active():
                    bbox = track.bbox
                    track_id = track.track_id
                    x1, y1 = int(bbox['bb_left']), int(bbox['bb_top'])
                    x2, y2 = x1 + int(bbox['bb_width']), y1 + int(bbox['bb_height'])
                    color = ((track_id * 50) % 255, (track_id * 100) % 255, (track_id * 150) % 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
    print("Saving tracking results...")
    save_tracking_results(all_frame_tracks, "ADL-Rundle-6_reid", output_txt_path)
    print("Tracking results saved.")
    print(f"Video saved to: {output_video_path}")
