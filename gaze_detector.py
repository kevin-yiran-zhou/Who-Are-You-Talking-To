import cv2
import mediapipe as mp
import numpy as np
import math

class GazeDetector:
    def __init__(self, show_pitch_yaw=False):
        # --- Configuration ---
        self.GAZE_THRESHOLD_YAW = 7
        self.GAZE_THRESHOLD_PITCH = 10
        # self.GAZE_THRESHOLD_YAW = 30
        # self.GAZE_THRESHOLD_PITCH = 30
        self.YAW_OFFSET = 54.5    # Your calibrated value
        self.PITCH_OFFSET = 87.0  # Your calibrated value
        self.WIDTH = 1280
        self.HEIGHT = 720
        self.TRACKING_THRESHOLD = 150
        self.show_pitch_yaw = show_pitch_yaw

        # --- Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=3, # Multi-user support
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        # Tracking State
        self.tracked_faces = []
        
        # 3D Model Points (Generic Face)
        self.face_3d = np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [0.0, -330.0, -65.0],       # Chin
            [-225.0, 170.0, -135.0],    # Left eye corner
            [225.0, 170.0, -135.0],     # Right eye corner
            [-150.0, -150.0, -125.0],   # Left Mouth corner
            [150.0, -150.0, -125.0]     # Right mouth corner
        ], dtype=np.float64)

    def _get_distance(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def process_frame(self):
        """
        Captures one frame, processes gaze, and returns the frame and a boolean status.
        Returns: (frame, is_someone_looking)
        """
        if not self.cap.isOpened():
            return None, False

        success, image = self.cap.read()
        if not success:
            return None, False

        img_h, img_w, img_c = image.shape
        
        # Pre-processing
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_frame_faces = []
        is_someone_looking = False # The master flag we will return

        if results.multi_face_landmarks:
            matched_tracked_indices = set()

            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                nose_2d = (0, 0)
                
                # Extract Landmarks
                landmark_indices = [1, 152, 33, 263, 61, 291]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in landmark_indices:
                        if idx == 1: nose_2d = (lm.x * img_w, lm.y * img_h)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])

                if len(face_2d) != 6: continue
                
                face_2d_ordered = np.array(face_2d, dtype=np.float64)

                # Camera Matrix
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # --- TRACKING LOGIC ---
                best_match_idx = -1
                min_dist = float('inf')

                for i, tracked_face in enumerate(self.tracked_faces):
                    if i in matched_tracked_indices: continue
                    dist = self._get_distance(nose_2d, tracked_face['center'])
                    if dist < min_dist and dist < self.TRACKING_THRESHOLD:
                        min_dist = dist
                        best_match_idx = i

                use_guess = best_match_idx != -1
                if use_guess: matched_tracked_indices.add(best_match_idx)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(
                    self.face_3d, face_2d_ordered, cam_matrix, dist_matrix,
                    useExtrinsicGuess=use_guess,
                    rvec=self.tracked_faces[best_match_idx]['rot_vec'] if use_guess else None,
                    tvec=self.tracked_faces[best_match_idx]['trans_vec'] if use_guess else None
                )

                # Get Angles
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch, yaw = angles[0], angles[1]

                # Logic & Visualization
                status_text = ""
                color = (0, 0, 255)

                if pitch < 0:
                    status_text = "INITIALIZING..."
                    color = (0, 165, 255)
                else:
                    # Valid pose - save for tracking
                    current_frame_faces.append({'center': nose_2d, 'rot_vec': rot_vec, 'trans_vec': trans_vec})
                    
                    yaw_centered = yaw - self.YAW_OFFSET
                    pitch_centered = pitch - self.PITCH_OFFSET
                    
                    if abs(yaw_centered) < self.GAZE_THRESHOLD_YAW and abs(pitch_centered) < self.GAZE_THRESHOLD_PITCH:
                        status_text = "LOOKING AT ROBOT"
                        color = (0, 255, 0)
                        is_someone_looking = True # Trigger response!
                    else:
                        status_text = "LOOKING AWAY"
                        color = (0, 0, 255)

                    # --- VISUALIZATION UPDATE ---
                    # Calculate nose position for text placement
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    
                    # Determine color for yaw/pitch values: red if exceeds threshold, green otherwise
                    exceeds_threshold = (abs(yaw_centered) >= self.GAZE_THRESHOLD_YAW or 
                                        abs(pitch_centered) >= self.GAZE_THRESHOLD_PITCH)
                    value_color = (0, 0, 255) if exceeds_threshold else (0, 255, 0)  # BGR: red or green
                    
                    # Format yaw and pitch text
                    yaw_text = f"Yaw: {round(yaw_centered)}"
                    pitch_text = f"Pitch: {round(pitch_centered)}"
                    
                    # Font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    line_height = 25
                    
                    # Draw status text
                    cv2.putText(image, status_text, (p1[0]-50, p1[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw yaw and pitch values below status text (if enabled)
                    if self.show_pitch_yaw:
                        cv2.putText(image, yaw_text, (p1[0]-50, p1[1]-5), font, font_scale, value_color, thickness)
                        cv2.putText(image, pitch_text, (p1[0]-50, p1[1]+line_height), font, font_scale, value_color, thickness)

        # Update tracking memory
        self.tracked_faces = current_frame_faces
        return image, is_someone_looking

    def release(self):
        self.cap.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()