import numpy as np
import os
from scipy.optimize import linear_sum_assignment
import cv2
import glob


db = "public-dataset"
def load_mot_detections(file_path):
    detections = {}

    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split(',')
            if len(fields) != 10:
                continue

            frame = int(fields[0])
            detection = {
                'frame' : frame,
                'id': int(fields[1]),
                'bb_left': float(fields[2]),
                'bb_top': float(fields[3]),
                'bb_width': float(fields[4]),
                'bb_height': float(fields[5]),
                'conf': float(fields[6]),
                'x': float(fields[7]),
                'y': float(fields[8]),
                'z': float(fields[9])
            }

            if frame not in detections:
                detections[frame] = []

            detections[frame].append(detection)

    return detections


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


def create_similarity_matrix(tracked_boxes, detected_boxes):
    if not tracked_boxes or not detected_boxes:
        return np.zeros((len(tracked_boxes), len(detected_boxes)))
    t_boxes = np.array([[t.bbox['bb_left'], t.bbox['bb_top'], 
                         t.bbox['bb_left']+t.bbox['bb_width'], t.bbox['bb_top']+t.bbox['bb_height']] 
                        for t in tracked_boxes])
    
    d_boxes = np.array([[d['bb_left'], d['bb_top'], 
                         d['bb_left']+d['bb_width'], d['bb_top']+d['bb_height']] 
                        for d in detected_boxes])

    xA = np.maximum(t_boxes[:, 0][:, None], d_boxes[:, 0])
    yA = np.maximum(t_boxes[:, 1][:, None], d_boxes[:, 1])
    xB = np.minimum(t_boxes[:, 2][:, None], d_boxes[:, 2])
    yB = np.minimum(t_boxes[:, 3][:, None], d_boxes[:, 3])

    inter_w = np.maximum(0, xB - xA)
    inter_h = np.maximum(0, yB - yA)
    inter_area = inter_w * inter_h

    area_t = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
    area_d = (d_boxes[:, 2] - d_boxes[:, 0]) * (d_boxes[:, 3] - d_boxes[:, 1])
    
    union_area = area_t[:, None] + area_d - inter_area
    
    return inter_area / (union_area + 1e-6)


def associate_detections_to_tracks(similarity_matrix, iou_threshold=0.3):
    if similarity_matrix.size == 0:
        return [], list(range(similarity_matrix.shape[0])), list(range(similarity_matrix.shape[1]))
    cost_matrix = 1 - similarity_matrix
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matched_indices = []
    unmatched_tracks = list(range(similarity_matrix.shape[0]))
    unmatched_detections = list(range(similarity_matrix.shape[1]))

    for t, d in zip(track_indices, detection_indices):
        if similarity_matrix[t, d] >= iou_threshold:
            matched_indices.append((t, d))
            unmatched_tracks.remove(t)
            unmatched_detections.remove(d)

    return matched_indices, unmatched_tracks, unmatched_detections


class Track:
    def __init__(self, track_id, bbox, max_missed_frames=5):
        self.track_id = track_id
        self.bbox = bbox
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames

    def update(self, bbox):
        self.bbox = bbox
        self.missed_frames = 0

    def increment_missed(self):
        self.missed_frames += 1

    def is_active(self):
        return self.missed_frames <= self.max_missed_frames


class Tracker:
    def __init__(self, max_missed_frames=5):
        self.tracks = []
        self.next_track_id = 0
        self.max_missed_frames = max_missed_frames

    def manage_tracks(self, matched, unmatched_tracks, unmatched_detections, detected_boxes):
        for track_idx, detection_idx in matched:
            if track_idx < len(self.tracks) and detection_idx < len(detected_boxes):
                self.tracks[track_idx].update(detected_boxes[detection_idx])
            else:
                print(f"Warning: Invalid match indices - track_idx: {track_idx}, detection_idx: {detection_idx}")

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].increment_missed()

        self.tracks = [track for track in self.tracks if track.is_active()]

        for detection_idx in unmatched_detections:
            new_track = Track(self.next_track_id, detected_boxes[detection_idx], self.max_missed_frames)
            self.tracks.append(new_track)
            self.next_track_id += 1


def display_and_save_tracking_results(video_path, tracker, output_path):
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for track in tracker.tracks:
            if track.is_active():
                bbox = track.bbox
                track_id = track.track_id
                x1, y1 = int(bbox['bb_left']), int(bbox['bb_top'])
                x2, y2 = x1 + int(bbox['bb_width']), y1 + int(bbox['bb_height'])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Tracking', frame)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


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

    print(f"Tracking results saved to {output_path}")


if __name__ == "__main__":
    base_folder = "ADL-Rundle-6"
    folder = os.path.join(base_folder, "det")
    img_folder = os.path.join(base_folder, "img1")
    bbox_file = os.path.join(folder, "public-dataset", "det.txt")
    print(f"Bounding box file exists: {os.path.exists(bbox_file)}")
    output_video_path = os.path.join(folder, "output_video.avi")
    output_txt_path = os.path.join(folder, "public-dataset")
    tracker = Tracker()
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
            print(f"Frame {frame_idx}: Detected boxes: {len(detected_boxes)}")
            similarity_matrix = create_similarity_matrix(tracker.tracks, detected_boxes)
            print(f"Frame {frame_idx}: Similarity matrix shape: {similarity_matrix.shape}")
            matched, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(similarity_matrix)
            print(f"Frame {frame_idx}: Matched: {len(matched)}, Unmatched Tracks: {len(unmatched_tracks)}, Unmatched Detections: {len(unmatched_detections)}")
            tracker.manage_tracks(matched, unmatched_tracks, unmatched_detections, detected_boxes)
            print(f"Frame {frame_idx}: Active tracks: {[track.track_id for track in tracker.tracks if track.is_active()]}")
            all_frame_tracks[frame_idx] = []
            for track in tracker.tracks:
                if track.is_active():
                    track_copy = Track(track.track_id, track.bbox.copy(), track.max_missed_frames)
                    all_frame_tracks[frame_idx].append(track_copy)

            for track in tracker.tracks:
                if track.is_active():
                    bbox = track.bbox
                    track_id = track.track_id
                    x1, y1 = int(bbox['bb_left']), int(bbox['bb_top'])
                    x2, y2 = x1 + int(bbox['bb_width']), y1 + int(bbox['bb_height'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cv2.destroyAllWindows()
    print("Saving tracking results...")
    save_tracking_results(all_frame_tracks, "ADL-Rundle-6", output_txt_path)
    print("Tracking results saved.")
