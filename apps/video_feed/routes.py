import cv2
from flask import render_template, request, Response, redirect, url_for, jsonify
from datetime import datetime
from apps.video_feed import blueprint
from apps.video_feed.confident_new import extract_landmarks, predict_with_confidence, pose, model, scaler, mp_drawing, mp_pose
from apps.config import API_GENERATOR
from collections import deque
import time
import numpy as np
import warnings
import pyttsx3

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# List of yoga poses and their instructions
yoga_poses_data = {
    "Chair": "Stand with your feet together, then bend your knees and lower your hips as if sitting in a chair. Raise your arms overhead.",
    "Cobra": "Lie face down, place your hands under your shoulders, then lift your chest while keeping your hips on the ground.",
    "Tree": "Stand on one leg, place the sole of your other foot on your inner thigh or calf, and bring your hands together at your chest.",
}

# Global variables for tracking
current_text = ""
reps = 0
last_feedback_time = 0
last_feedback_confidence = 0
high_confidence_spoken = False
pose_tracker = deque(maxlen=15)
current_pose = None
pose_start_time = None
pose_hold_time = 0
rep_count = 0
alpha = 0.3
smoothed_landmarks = None
CONFIDENCE_THRESHOLD = 0.7
POSE_HOLD_THRESHOLD = 5
last_audio_time = 0
AUDIO_COOLDOWN = 5
confidence_window = deque(maxlen=10)
noProgress = 0
last_detected_pose = None

@blueprint.route('/get_text')
def get_text():
    return jsonify({'text': current_text})

def update_text(text):
    global current_text
    current_text = text

@blueprint.route('/get_reps')
def get_reps():
    return jsonify({'text': reps})

def update_reps(n):
    global reps
    reps = n

@blueprint.route('/interface/<int:pose_index>')
def analyze(pose_index):
    pose_data = [
        {"name": "Tree Pose (Vrikshasana)", "level": "Beginner", "pose_key": "Tree"},
        {"name": "Chair Pose (Bhujangasana)", "level": "Beginner", "pose_key": "Chair"},
        {"name": "Cobra Pose (Utkatasana)", "level": "Intermediate", "pose_key": "Cobra"},
    ]
    
    pose = pose_data[pose_index]
    next_pose_index = pose_index + 1

    return render_template(
        'home/interface.html',
        segment='interface', 
        API_GENERATOR=len(API_GENERATOR), 
        show_sideBar=False,
        show_nav=False,
        pose=pose, 
        next_pose_index=next_pose_index
    )

def generate_frames(target):
    cap = cv2.VideoCapture(1)
    camera_on = True
    global smoothed_landmarks, pose_start_time, rep_count, last_audio_time
    global last_feedback_time, last_feedback_confidence, high_confidence_spoken, noProgress
    
    speak(yoga_poses_data[target])

    while camera_on:
        ret, frame = cap.read()
        if not ret:
            update_text("Camera feed failed. Restarting the camera.")
            cap.release()
            cap = cv2.VideoCapture(1)
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        landmarks = extract_landmarks(results)

        confidence = 0
        weighted_confidence = 0
        
        if landmarks is not None:
            if smoothed_landmarks is None:
                smoothed_landmarks = landmarks
            else:
                smoothed_landmarks = alpha * landmarks + (1 - alpha) * smoothed_landmarks
            
            prediction, confidence = predict_with_confidence(model, scaler, landmarks)
            confidence_window.append(confidence)
            weighted_confidence = np.average(confidence_window, weights=range(1, len(confidence_window) + 1))

            if weighted_confidence > CONFIDENCE_THRESHOLD:
                pose_tracker.append(prediction)

                if len(pose_tracker) == pose_tracker.maxlen:
                    current_pose = max(set(pose_tracker), key=pose_tracker.count)
                    matched_pose = False
                    
                    for yoga_pose in yoga_poses_data.keys():
                        if current_pose.lower() in yoga_pose.lower():
                            current_pose = yoga_pose
                            matched_pose = True
                            break
                            
                    if not matched_pose:
                        current_pose = "Unknown Pose"
                    
                    if target.lower() in current_pose.lower():
                        update_text(f"Target pose detected: {current_pose}")
                        
                        cv2.putText(image, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, f"Confidence: {weighted_confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        current_time = time.time()
                        if current_pose in yoga_poses_data and weighted_confidence > CONFIDENCE_THRESHOLD:
                            if pose_start_time is None:
                                pose_start_time = current_time
                            pose_hold_time = current_time - pose_start_time
                            cv2.putText(image, f"Hold Time: {pose_hold_time:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            if current_time - last_audio_time > AUDIO_COOLDOWN:
                                if pose_hold_time < POSE_HOLD_THRESHOLD:
                                    update_text(f"Continue holding the {current_pose} for {POSE_HOLD_THRESHOLD - pose_hold_time:.0f} more seconds")
                                else:
                                    update_text(f"Excellent! You've held the {current_pose} successfully")
                                last_audio_time = current_time

                            if pose_hold_time > POSE_HOLD_THRESHOLD:
                                rep_count += 1
                                pose_start_time = None
                                update_text(f"Rep {rep_count} completed")
                                update_reps(rep_count)
                                
                        cv2.putText(image, f"Reps: {rep_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        update_text(f"Detected pose '{current_pose}' does not match target '{target}'. Ignoring...")
                        cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Feedback mechanism
        current_time = time.time()
        if weighted_confidence <= 0.5:
            if current_time - last_feedback_time >= 5:
                noProgress += 1
                if noProgress == 20:
                    speak(yoga_poses_data[target])
                    update_text("pose not right 20 times")
                else:
                    update_text(f"Pose not right. Try again Please. {noProgress}")
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
                high_confidence_spoken = False
        elif 0.5 < weighted_confidence < 0.8:
            if current_time - last_feedback_time >= 5:
                update_text("Good going. Keep trying!")
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence
                high_confidence_spoken = False
        else:
            if not high_confidence_spoken:
                update_text("Congrats! You've got how to do this pose now.")
                speak("Congrats! You've got how to do this pose now.")
                high_confidence_spoken = True
                last_feedback_time = current_time
                last_feedback_confidence = weighted_confidence

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        flag, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@blueprint.route('/video_feed/<target>')
def video_feed(target):
    return Response(generate_frames(target), mimetype='multipart/x-mixed-replace; boundary=frame')

@blueprint.route('/stop_video')
def stop_video():
    try:
        return redirect(url_for('home_blueprint.index'))
    except Exception as e:
        print(f"error while redirecting to home: {e}")


# In your video_feed_blueprint.py or routes.py

# from flask import Response, Blueprint
# import cv2

# video_feed_blueprint = Blueprint('video_feed_blueprint', __name__)

# def generate_frames():
#     camera = cv2.VideoCapture(0)  # Use 0 for default webcam
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @video_feed_blueprint.route('/video_feed/<target>')
# def video_feed(target):
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @video_feed_blueprint.route('/analyze/<target>')
# def analyze(target):
#     # Your analysis logic here
#     return {'success': True}

# @video_feed_blueprint.route('/stop_video')
# def stop_video():
#     # Your stop video logic here
#     return {'success': True}

# @video_feed_blueprint.route('/post_session', methods=['POST'])
# def post_session():
#     # Your session posting logic here
#     return {'success': True}