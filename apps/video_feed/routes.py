import cv2
from flask import render_template, request, Response, redirect, url_for, jsonify, session
from flask_login import current_user
from apps.video_feed.dbmodels import YogaSession, HeartRateData, YogaPoseData
from datetime import datetime, timezone
from apps import db
from apps.video_feed.confident_new import extract_landmarks, predict_with_confidence, pose, model, scaler, mp_drawing, mp_pose
from apps.video_feed import blueprint
from apps.config import API_GENERATOR
from collections import deque
import time
import numpy as np
import warnings
import pyttsx3
from sqlalchemy import func

# Global variables for state management
current_text = "Initializing..."
reps = 0
camera_index = 0 # Default camera index
global_cap = None

# Pose tracking variables
pose_tracker = deque(maxlen=15)
current_pose = None
pose_start_time = None
pose_hold_time = 0
rep_count = 0
last_feedback_time = 0
last_feedback_confidence = 0
high_confidence_spoken = False
noProgress = 0
last_detected_pose = None
smoothed_landmarks = None

# Constants
CONFIDENCE_THRESHOLD = 0.7
POSE_HOLD_THRESHOLD = 5
AUDIO_COOLDOWN = 5
alpha = 0.3  # Smoothing factor

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Yoga poses data
yoga_poses_data = {
    "Chair": "Stand with your feet together, then bend your knees and lower your hips as if sitting in a chair. Raise your arms overhead.",
    "Cobra": "Lie face down, place your hands under your shoulders, then lift your chest while keeping your hips on the ground.",
    "Tree": "Stand on one leg, place the sole of your other foot on your inner thigh or calf, and bring your hands together at your chest.",
}

def init_camera():
    """Initialize the camera and return the capture object"""
    global global_cap
    if global_cap is not None:
        global_cap.release()
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise Exception("Failed to open camera")
        return cap
    except Exception as e:
        print(f"Camera initialization error: {e}")
        return None

def speak(text):
    """Text-to-speech function"""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech synthesis error: {e}")

# Route handlers
@blueprint.route('/get_text')
def get_text():
    global current_text
    return jsonify({'text': current_text})

def update_text(text):
    global current_text
    current_text = text
    print(f"Updated text: {text}")  # Debug print

@blueprint.route('/get_reps')
def get_reps():
    global reps
    return jsonify({'text': reps})

def update_reps(n):
    global reps
    reps = n
    print(f"Updated reps: {n}")  # Debug print

@blueprint.route('/interface/<int:pose_index>')
def analyze(pose_index):
    pose_data = [
        {"name": "Tree Pose (Vrikshasana)", "level": "Beginner", "pose_key": "Tree"},
        {"name": "Chair Pose (Utkatasana)", "level": "Beginner", "pose_key": "Chair"},
        {"name": "Cobra Pose (Bhujangasana)", "level": "Intermediate", "pose_key": "Cobra"},
    ]
    
    if pose_index >= len(pose_data):
        return "Invalid pose index", 404
        
    pose = pose_data[pose_index]
    next_pose_index = (pose_index + 1) % len(pose_data)

    return render_template(
        'home/interface.html',
        segment='interface', 
        API_GENERATOR=len(API_GENERATOR), 
        show_sideBar=False,
        show_nav=False,
        pose=pose, 
        next_pose_index=next_pose_index
    )

def generate_frames(target_pose):
    print(f"Starting video feed for target: {target_pose}")
    cap = cv2.VideoCapture(0)  # Use the appropriate index

    if not cap.isOpened():
        print("Camera initialization error: Failed to open camera")
        raise Exception("Camera initialization failed")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Add text overlay for debugging
            cv2.putText(frame, f"Pose: {target_pose}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Failed to encode frame.")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        print(f"Starting video capture for {target} pose")
        speak(yoga_poses_data[target])
        
        confidence_window = deque(maxlen=10)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                update_text("Camera feed failed. Restarting...")
                cap = init_camera()
                continue

            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            landmarks = extract_landmarks(results)
            
            if landmarks is not None:
                # Smoothing
                if smoothed_landmarks is None:
                    smoothed_landmarks = landmarks
                else:
                    smoothed_landmarks = alpha * landmarks + (1 - alpha) * smoothed_landmarks

                # Prediction
                prediction, confidence = predict_with_confidence(model, scaler, landmarks)
                confidence_window.append(confidence)
                weighted_confidence = np.average(confidence_window, weights=range(1, len(confidence_window) + 1))

                # Pose tracking
                if weighted_confidence > CONFIDENCE_THRESHOLD:
                    pose_tracker.append(prediction)

                    if len(pose_tracker) == pose_tracker.maxlen:
                        current_pose = max(set(pose_tracker), key=pose_tracker.count)
                        
                        # Match detected pose with target
                        matched_pose = False
                        for yoga_pose in yoga_poses_data.keys():
                            if current_pose.lower() in yoga_pose.lower():
                                current_pose = yoga_pose
                                matched_pose = True
                                break
                                
                        if not matched_pose:
                            current_pose = "Unknown Pose"

                        # Process matched pose
                        if target.lower() in current_pose.lower():
                            update_text(f"Target pose detected: {current_pose}")
                            
                            # Display info on frame
                            cv2.putText(image, f"Pose: {current_pose}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(image, f"Confidence: {weighted_confidence:.2f}", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Timer and rep counting
                            current_time = time.time()
                            if weighted_confidence > CONFIDENCE_THRESHOLD:
                                if pose_start_time is None:
                                    pose_start_time = current_time
                                pose_hold_time = current_time - pose_start_time
                                
                                cv2.putText(image, f"Hold Time: {pose_hold_time:.2f}s", 
                                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                                # Audio feedback
                                if current_time - last_feedback_time > AUDIO_COOLDOWN:
                                    if pose_hold_time < POSE_HOLD_THRESHOLD:
                                        update_text(f"Continue holding for {POSE_HOLD_THRESHOLD - pose_hold_time:.0f} more seconds")
                                    else:
                                        update_text("Excellent! Hold completed")
                                    last_feedback_time = current_time

                                # Rep counting
                                if pose_hold_time > POSE_HOLD_THRESHOLD:
                                    rep_count += 1
                                    pose_start_time = None
                                    update_text(f"Rep {rep_count} completed")
                                    update_reps(rep_count)
                            
                            cv2.putText(image, f"Reps: {rep_count}", (10, 150), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            update_text(f"Detected '{current_pose}' - not matching target '{target}'")
                            cv2.putText(image, "Detecting...", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Confidence-based feedback
                if weighted_confidence <= 0.5:
                    if current_time - last_feedback_time >= 5:
                        noProgress += 1
                        if noProgress == 20:
                            speak(yoga_poses_data[target])
                            update_text("Pose not correct after 20 attempts")
                        else:
                            update_text(f"Please try again. Attempt {noProgress}")
                        last_feedback_time = current_time
                        last_feedback_confidence = weighted_confidence
                        high_confidence_spoken = False
                elif 0.5 < weighted_confidence < 0.8:
                    if current_time - last_feedback_time >= 5:
                        update_text("Good progress! Keep improving")
                        last_feedback_time = current_time
                        last_feedback_confidence = weighted_confidence
                else:  # weighted_confidence >= 0.8
                    if not high_confidence_spoken:
                        update_text("Excellent form!")
                        speak("Excellent form!")
                        high_confidence_spoken = True
                        last_feedback_time = current_time
                        last_feedback_confidence = weighted_confidence

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            # Convert frame to bytes
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        if cap:
            cap.release()

@blueprint.route('/video_feed/<target>')
def video_feed(target):
    """Video streaming route"""
    print(f"Starting video feed for target: {target}")
    return Response(generate_frames(target),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@blueprint.route('/stop_video')
def stop_video():
    """Stop video streaming"""
    global global_cap
    try:
        if global_cap:
            global_cap.release()
            global_cap = None
        return redirect(url_for('home_blueprint.index'))
    except Exception as e:
        print(f"Error stopping video: {e}")
        return str(e), 500

