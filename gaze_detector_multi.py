import cv2
import mediapipe as mp
import numpy as np
import math

# --- Constants ---
GAZE_THRESHOLD_YAW = 10
GAZE_THRESHOLD_PITCH = 10

# --- CALIBRATION ---
YAW_OFFSET = 54.5
PITCH_OFFSET = 87.0
# ------------------

# --- Resolution Request ---
WIDTH = 1280
HEIGHT = 720

# --- Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

face_mesh = mp_face_mesh.FaceMesh(
    # --- MULTI-TRACK --- Look for more than one face
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Error: Cannot open camera. Exiting.")
    exit()

# --- MULTI-TRACK ---
# This list will be our "memory" of tracked faces
# It will store dictionaries like:
# {'center': (x, y), 'rot_vec': rvec, 'trans_vec': tvec}
tracked_faces = []
# How close a new face must be to an old face to be a "match" (in pixels)
TRACKING_THRESHOLD = 150 

# Simple distance calculator
def get_distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

print("Starting Multi-Face Gaze Detector... Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    img_h, img_w, img_c = image.shape
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # --- MULTI-TRACK ---
    # This list will hold the faces we find *in this frame*
    # We will replace the old 'tracked_faces' list at the end
    current_frame_faces = []
    
    if results.multi_face_landmarks:
        # Keep track of which old faces have been matched
        matched_tracked_indices = set()

        for face_landmarks in results.multi_face_landmarks:
            
            face_3d = []
            face_2d = []
            nose_2d = (0, 0) # Will be overwritten
            
            landmark_indices = [1, 152, 33, 263, 61, 291]
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in landmark_indices:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])

            if len(face_2d) == 6:
                face_2d_ordered = np.array([
                    face_2d[0], face_2d[1], face_2d[2], 
                    face_2d[3], face_2d[4], face_2d[5]
                ], dtype=np.float64)
            else:
                continue 
            
            face_3d = np.array([
                [0.0, 0.0, 0.0],            # Nose tip
                [0.0, -330.0, -65.0],       # Chin
                [-225.0, 170.0, -135.0],    # Left eye corner
                [225.0, 170.0, -135.0],     # Right eye corner
                [-150.0, -150.0, -125.0],   # Left Mouth corner
                [150.0, -150.0, -125.0]     # Right mouth corner
            ], dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1] ])
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # --- MULTI-TRACK ---
            # Find the closest "old" face to this "new" face
            best_match_idx = -1
            min_dist = float('inf')

            for i, tracked_face in enumerate(tracked_faces):
                if i in matched_tracked_indices:
                    continue # This old face is already matched, skip it
                
                dist = get_distance(nose_2d, tracked_face['center'])
                
                if dist < min_dist and dist < TRACKING_THRESHOLD:
                    min_dist = dist
                    best_match_idx = i

            # --- MULTI-TRACK ---
            # Get the pose hint (if we have one)
            use_guess = best_match_idx != -1
            rvec_guess = tracked_faces[best_match_idx]['rot_vec'] if use_guess else None
            tvec_guess = tracked_faces[best_match_idx]['trans_vec'] if use_guess else None
            
            if use_guess:
                matched_tracked_indices.add(best_match_idx)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, 
                face_2d_ordered, 
                cam_matrix, 
                dist_matrix,
                useExtrinsicGuess=use_guess,
                rvec=rvec_guess,
                tvec=tvec_guess
            )
            
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            yaw = angles[1]
            pitch = angles[0]
            
            # --- MULTI-TRACK ---
            # We now store all visualization data per-face
            face_data_for_drawing = {}

            if pitch < 0:
                # BAD POSE
                gaze_status = "INITIALIZING..."
                status_color = (0, 165, 255) # Orange
                
                # We won't store rot_vec, so the "guess" will fail
                # next frame, forcing a fresh calculation
                face_data_for_drawing = {
                    'status': gaze_status,
                    'color': status_color,
                    'p1': (int(nose_2d[0]), int(nose_2d[1]))
                }
            else:
                # GOOD POSE
                # Store this pose for the next frame
                current_frame_faces.append({
                    'center': nose_2d,
                    'rot_vec': rot_vec,
                    'trans_vec': trans_vec
                })

                yaw_centered = yaw - YAW_OFFSET
                pitch_centered = pitch - PITCH_OFFSET
                
                if abs(yaw_centered) < GAZE_THRESHOLD_YAW and abs(pitch_centered) < GAZE_THRESHOLD_PITCH:
                    gaze_status = "LOOKING AT ROBOT"
                    status_color = (0, 255, 0) # Green
                else:
                    gaze_status = "LOOKING AWAY"
                    status_color = (0, 0, 255) # Red

                # --- Visualization ---
                (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                
                face_data_for_drawing = {
                    'status': gaze_status,
                    'color': status_color,
                    'p1': p1,
                    'p2': p2,
                    'yaw': yaw_centered,
                    'pitch': pitch_centered
                }
            
            # --- MULTI-TRACK ---
            # Draw all the info for *this* face
            cv2.putText(image, face_data_for_drawing['status'], 
                        (face_data_for_drawing['p1'][0] - 80, face_data_for_drawing['p1'][1] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_data_for_drawing['color'], 2)

            if 'p2' in face_data_for_drawing:
                # Only draw line and angles if we have a good pose
                # cv2.line(image, face_data_for_drawing['p1'], face_data_for_drawing['p2'], (255, 0, 0), 2)
                cv2.putText(image, f"Y: {face_data_for_drawing['yaw']:.1f}", 
                            (face_data_for_drawing['p1'][0] + 20, face_data_for_drawing['p1'][1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(image, f"P: {face_data_for_drawing['pitch']:.1f}", 
                            (face_data_for_drawing['p1'][0] + 20, face_data_for_drawing['p1'][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # --- MULTI-TRACK ---
    # At the end of the frame, update our "memory"
    # This automatically handles faces leaving the frame
    tracked_faces = current_frame_faces
    
    cv2.imshow('Multi-Face Gaze Detector', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
print("Gaze detector stopped.")