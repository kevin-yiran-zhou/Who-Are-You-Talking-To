import cv2
import mediapipe as mp
import numpy as np
import math

# --- Constants ---
GAZE_THRESHOLD_YAW = 20
GAZE_THRESHOLD_PITCH = 25

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
    max_num_faces=1,
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

# We'll store the previous pose to prevent jitter
prev_rot_vec = None
prev_trans_vec = None

print("Starting Gaze Detector... Press 'q' to quit.")

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

    gaze_status = "LOOKING AWAY"
    status_color = (0, 0, 255) # Red

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            face_3d = []
            face_2d = []
            
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

            # Use the previous pose if we have one
            use_guess = prev_rot_vec is not None

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, 
                face_2d_ordered, 
                cam_matrix, 
                dist_matrix,
                useExtrinsicGuess=use_guess,
                rvec=prev_rot_vec if use_guess else None,
                tvec=prev_trans_vec if use_guess else None
            )
            
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            yaw = angles[1]
            pitch = angles[0]
            roll = angles[2]

            # --- THIS IS THE FIX ---
            # We know the "good" pose has a raw pitch > 0 (around +70-80)
            # The "bad" flipped pose has a raw pitch < 0 (around -90)
            # This is our "sanity check"
            if pitch < 0:
                # BAD POSE DETECTED!
                # Do not store this pose. This will force the
                # next frame to re-calculate from scratch.
                prev_rot_vec = None
                prev_trans_vec = None
                
                # We can also just skip drawing this bad frame
                gaze_status = "INITIALIZING..."
                status_color = (0, 165, 255) # Orange
                
            else:
                # GOOD POSE
                # Store it for a stable hint next frame
                prev_rot_vec = rot_vec
                prev_trans_vec = trans_vec

                # Now do the calibration and status
                yaw_centered = yaw - YAW_OFFSET
                pitch_centered = pitch - PITCH_OFFSET
                
                if abs(yaw_centered) < GAZE_THRESHOLD_YAW and abs(pitch_centered) < GAZE_THRESHOLD_PITCH:
                    gaze_status = "LOOKING AT ROBOT"
                    status_color = (0, 255, 0) # Green

                # --- Visualization ---
                (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(image, p1, p2, (255, 0, 0), 2)
                
                cv2.putText(image, f"Yaw (Cal): {yaw_centered:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(image, f"Pitch (Cal): {pitch_centered:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    else:
        # If no face is detected, reset the pose hint
        prev_rot_vec = None
        prev_trans_vec = None

    # Display the final Gaze Status
    cv2.rectangle(image, (0, 0), (img_w, 40), (0,0,0), -1)
    cv2.putText(image, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    cv2.imshow('Gaze Detection Module', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
print("Gaze detector stopped.")