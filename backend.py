import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Clothing images dictionary
clothing_images = {
    "shirt1": "static/shirts/shirt1.png",
    "shirt2": "static/shirts/shirt2.png",
    "shirt3": "static/shirts/shirt3.png",
    "shirt4": "static/shirts/shirt4.png",
    "shirt5": "static/shirts/shirt5.png",
    "shirt6": "static/shirts/shirt6.png",
    "shirt7": "static/shirts/shirt7.png",
    "shirt8": "static/shirts/shirt8.png",
    "shirt9": "static/shirts/shirt9.png",
    "shirt10": "static/shirts/shirt10.png",
    "shirt11": "static/shirts/shirt11.png",
    "shirt12": "static/shirts/shirt12.png", 
    "shirt13":"static/shirts/shirt13.png",


    "dress1": "static/women/dress1.png",
    "dress2": "static/women/dress2.png",
    "dress3": "static/women/dress3.png",
    "dress4": "static/women/dress4.png",
    "dress5": "static/women/dress5.png",
    "dress6": "static/women/dress6.png",
    "dress7": "static/women/dress7.png",
    "dress8": "static/women/dress8.png",
    "dress9": "static/women/dress9.png",
    "dress10": "static/women/dress10.png",
    "dress11": "static/women/dress11.png",
    "dress12": "static/women/dress12.png",
    "dress13": "static/women/dress13.png",
    "dress14": "static/women/dress14.png",
}

# Size scaling factors and button positions
size_factors = {"XS": 0.8, "S": 0.9, "M": 1.0, "L": 1.1, "XL": 1.2, "XXL": 1.3, "XXXL": 1.4}
button_y_positions = {"XS": 50, "S": 100, "M": 150, "L": 200, "XL": 250, "XXL": 300, "XXXL": 350}
button_radius = 40
button_x = 50
current_size = "M"

def load_clothing_image(clothing_id):
    try:
        clothing_path = clothing_images.get(clothing_id)
        clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)
        if clothing_img is None:
            raise FileNotFoundError(f"Clothing image {clothing_path} not found!")
        original_height, original_width = clothing_img.shape[:2]
        aspect_ratio = original_height / original_width
        print(f"Clothing {clothing_id} loaded. Shape: {clothing_img.shape}, Aspect Ratio: {aspect_ratio}")
        return clothing_img, aspect_ratio
    except Exception as e:
        print(f"Error loading clothing image: {e}")
        exit()

def overlay_clothing(frame, clothing_img, landmarks, size_factor, aspect_ratio, clothing_id):
    try:
        # Get shoulder and hip landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate shoulder width and torso height
        shoulder_width = abs(right_shoulder.x - left_shoulder.x) * frame.shape[1]
        torso_height = abs(left_shoulder.y - left_hip.y) * frame.shape[0]

        # Default parameters
        scaling_factor = 1.8
        vertical_offset = 0
        apply_rotation = False
        flip_after_rotation = False
        expansion_factor = 1.0

        # Clothing-specific adjustments
        if clothing_id.startswith("shirt"):
            if clothing_id == "shirt1":
                scaling_factor = 1.8
                vertical_offset = -30
            elif clothing_id == "shirt2":
                scaling_factor = 2.0
                vertical_offset = -45
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "shirt3":
                scaling_factor = 2.0
                vertical_offset = -60
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "shirt4":
                scaling_factor = 2.0
                vertical_offset = -100
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "shirt5":
                scaling_factor = 2.0
                vertical_offset = -100
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
        elif clothing_id.startswith("dress"):
            if clothing_id == "dress1":
                scaling_factor = 2.2
                vertical_offset = -40  # Adjusted from 15 to 5 to position closer to the neck
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "dress2":
                scaling_factor = 2.1
                vertical_offset = -45
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "dress3":
                scaling_factor = 2.2
                vertical_offset = -50
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "dress4":
                scaling_factor = 2.2
                vertical_offset = -50
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6
            elif clothing_id == "dress5":
                scaling_factor = 2.2
                vertical_offset = -50
                if current_size in ["L", "XL"]:
                    expansion_factor = 1.2
                elif current_size == "XXL":
                    expansion_factor = 1.4
                elif current_size == "XXXL":
                    expansion_factor = 1.6

        # Calculate clothing dimensions with expansion factor
        clothing_width = int(shoulder_width * scaling_factor * size_factor * expansion_factor)
        clothing_height = int(clothing_width * aspect_ratio)

        # Ensure clothing height fits the torso, adjusted for expansion
        torso_height_pixels = int(torso_height * 2.0 * expansion_factor)
        if clothing_height > torso_height_pixels:
            clothing_height = min(torso_height_pixels, clothing_height)
            clothing_width = int(clothing_height / aspect_ratio)

        # Calculate position
        center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
        top_y = int(min(left_shoulder.y, right_shoulder.y) * frame.shape[0])
        top_left_x = center_x - clothing_width // 2
        top_left_y = top_y + vertical_offset

        # Debug output
        print(f"Current size: {current_size}")
        # print(f"Left Shoulder: ({left_shoulder.x}, {left_shoulder.y})")
        # print(f"Right Shoulder: ({right_shoulder.x}, {right_shoulder.y})")
        # print(f"Left Hip: ({left_hip.x}, {left_hip.y})")
        # print(f"Right Hip: ({right_hip.x}, {right_hip.y})")
        # print(f"Shoulder Width: {shoulder_width:.2f} pixels")
        # print(f"Torso Height: {torso_height:.2f} pixels")
        # print(f"Expansion Factor: {expansion_factor}")
        # print(f"Torso Height Pixels (before cap): {torso_height_pixels}")
        # print(f"Resized Clothing: Width={clothing_width}, Height={clothing_height}")
        # print(f"Initial Clothing Position: Top-Left=({top_left_x}, {top_left_y})")

        # Resize clothing
        resized_clothing = cv2.resize(clothing_img, (clothing_width, clothing_height), interpolation=cv2.INTER_AREA)

        # Clip clothing if it exceeds frame boundaries
        if top_left_y + clothing_height > frame.shape[0]:
            excess = (top_left_y + clothing_height) - frame.shape[0]
            clothing_height = max(0, clothing_height - excess)
            resized_clothing = resized_clothing[:clothing_height, :]
            print(f"Clipped height due to bottom boundary. New height: {clothing_height}")
        if top_left_x + clothing_width > frame.shape[1]:
            excess = (top_left_x + clothing_width) - frame.shape[1]
            clothing_width = max(0, clothing_width - excess)
            resized_clothing = resized_clothing[:, :clothing_width]
            print(f"Clipped width due to right boundary. New width: {clothing_width}")
        if top_left_y < 0:
            excess = -top_left_y
            resized_clothing = resized_clothing[excess:, :]
            clothing_height -= excess
            top_left_y = 0
            print(f"Clipped height due to top boundary. New top_y: {top_left_y}")
        if top_left_x < 0:
            excess = -top_left_x
            resized_clothing = resized_clothing[:, excess:]
            clothing_width -= excess
            top_left_x = 0
            print(f"Clipped width due to left boundary. New top_x: {top_left_x}")

        # Extract alpha channel for transparency
        if resized_clothing.shape[2] == 4 and clothing_height > 0 and clothing_width > 0:
            clothing_rgb = resized_clothing[:, :, :3]
            alpha_mask = resized_clothing[:, :, 3] / 255.0
        else:
            print(f"Warning: Clothing {clothing_id} has no alpha channel or invalid dimensions! Shape: {resized_clothing.shape}")
            clothing_rgb = resized_clothing
            alpha_mask = np.ones((clothing_height, clothing_width)) if clothing_height > 0 and clothing_width > 0 else np.zeros((1, 1))

        # Overlay the clothing on the frame
        if clothing_height > 0 and clothing_width > 0 and top_left_y + clothing_height <= frame.shape[0] and top_left_x + clothing_width <= frame.shape[1]:
            roi = frame[top_left_y:top_left_y + clothing_height, top_left_x:top_left_x + clothing_width]
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + clothing_rgb[:, :, c] * alpha_mask
            frame[top_left_y:top_left_y + clothing_height, top_left_x:top_left_x + clothing_width] = roi
            print(f"Clothing {clothing_id} overlay successful! Final Position: Top-Left=({top_left_x}, {top_left_y}), Size: {clothing_width}x{clothing_height}")
        else:
            print(f"Warning: Clothing {clothing_id} not overlaid due to invalid position or size. Final Position: ({top_left_x}, {top_left_y}), Size: {clothing_width}x{clothing_height}")

        return frame
    except Exception as e:
        print(f"Error in overlay_clothing for {clothing_id}: {e}")
        return frame

@app.route('/start_tryon', methods=['POST'])
def start_tryon():
    data = request.get_json()
    print("Received request:", data)
    clothing_id = data.get('shirt_id')
    if clothing_id not in clothing_images:
        print(f"Invalid clothing ID: {clothing_id}")
        return jsonify({"error": "Invalid clothing ID"}), 400

    clothing_img, aspect_ratio = load_clothing_image(clothing_id)
    print(f"Starting try-on for {clothing_id}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500

    # Set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera opened successfully")

    global current_size
    window_name = f"Virtual Fitting Room - {clothing_id}"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame = overlay_clothing(frame, clothing_img, pose_results.pose_landmarks.landmark, size_factors[current_size], aspect_ratio, clothing_id)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_x = int(index_finger_tip.x * frame.shape[1])
                finger_y = int(index_finger_tip.y * frame.shape[0])

                for size, y_pos in button_y_positions.items():
                    distance = np.sqrt((finger_x - button_x) ** 2 + (finger_y - y_pos) ** 2)
                    if distance < button_radius:
                        current_size = size
                        print(f"Size selected: {current_size}")
                        break
        else:
            print("No hand landmarks detected!")

        for size, y_pos in button_y_positions.items():
            color = (0, 255, 0) if size == current_size else (255, 255, 0)
            cv2.circle(frame, (button_x, y_pos), button_radius, color, -1)
            cv2.putText(frame, size, (button_x - 15, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow(f"Virtual Fitting Room - {clothing_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Try-on for {clothing_id} completed")
    return jsonify({"status": "Try-on completed"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)