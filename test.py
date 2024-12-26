import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time

# Load the saved model and scaler
with open(r'C:\Users\pc\Desktop\gym\Yoga-app-master\Yoga-app-master\yoga_pose_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open(r'C:\Users\pc\Desktop\gym\Yoga-app-master\Yoga-app-master\yoga_pose_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# List of yoga poses and their instructions
yoga_poses = {
    "Chair Pose (Utkatasana)": "Stand with your feet together, then bend your knees and lower your hips as if sitting in a chair. Raise your arms overhead.",
    "Tree Pose (Vrikshasana)": "Stand on one leg, place the sole of your other foot on your inner thigh or calf, and bring your hands together at your chest."
}

def extract_landmarks(results):
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Pose tracking
pose_tracker = deque(maxlen=15)
current_pose = None
pose_start_time = None
pose_hold_time = 0
rep_count = 0

# Smoothing filter
alpha = 0.3  # Adjusted for more responsive smoothing
smoothed_landmarks = None

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.8  # Increased for more reliable predictions

# Pose hold time threshold
POSE_HOLD_THRESHOLD = 5  # Increased to 5 seconds for a more challenging hold

# Last detected pose
last_detected_pose = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect poses
    results = pose.process(image)
    
    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Extract landmarks
    landmarks = extract_landmarks(results)
    
    if landmarks is not None:
        # Apply smoothing filter
        if smoothed_landmarks is None:
            smoothed_landmarks = landmarks
        else:
            smoothed_landmarks = alpha * landmarks + (1 - alpha) * smoothed_landmarks
        
        # Scale the landmarks
        landmarks_scaled = scaler.transform([smoothed_landmarks])
        
        # Make prediction
        prediction = model.predict(landmarks_scaled)[0]
        confidence = np.max(model.predict_proba(landmarks_scaled))
        
        # Update pose tracker
        if confidence > CONFIDENCE_THRESHOLD:
            pose_tracker.append(prediction)
            if len(pose_tracker) == pose_tracker.maxlen:
                current_pose = max(set(pose_tracker), key=pose_tracker.count)
                # Map the detected pose to one of the yoga poses or handle 'no_pose'
                matched_pose = False
                for yoga_pose in yoga_poses.keys():
                    if current_pose.lower() in yoga_pose.lower():
                        current_pose = yoga_pose
                        matched_pose = True
                        break
                if not matched_pose:
                    current_pose = "Unknown Pose"

        # Display feedback based on confidence
        feedback_text = ""
        if confidence <= 0.5:
            feedback_text = "Pose not right. Try again Please."
        elif 0.5 < confidence < 0.8:
            feedback_text = "Good going. Keep trying!"
        else:
            feedback_text = "Excellent form!"

        # Display the prediction, confidence, and feedback
        cv2.putText(image, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, feedback_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Calculate and display angles (example for right elbow)
        if results.pose_landmarks:
            right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            cv2.putText(image, f"Right Elbow Angle: {angle:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Timer for pose hold
        current_time = time.time()
        if current_pose in yoga_poses and confidence > CONFIDENCE_THRESHOLD:
            if pose_start_time is None:
                pose_start_time = current_time
            pose_hold_time = current_time - pose_start_time
            cv2.putText(image, f"Hold Time: {pose_hold_time:.2f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update rep count
            if pose_hold_time > POSE_HOLD_THRESHOLD:
                rep_count += 1
                pose_start_time = None  # Reset timer
        else:
            pose_start_time = None
            pose_hold_time = 0
            
        cv2.putText(image, f"Reps: {rep_count}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    # Display the image
    cv2.imshow('Yoga Pose Classification', image)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()