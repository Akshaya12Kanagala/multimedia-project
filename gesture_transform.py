import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Display all files in the current directory to help user select an image
print("Files in current directory:")
for file in os.listdir():
    print("-", file)

# Prompt user to enter the image filename
image_name = input("Enter image filename (e.g., image.jpeg): ").strip()
image = cv2.imread(image_name)
if image is None:
    print("❌ Error: Image not found.")
    exit()

# Store original image dimensions
img_h, img_w = image.shape[:2]

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("❌ Error: Cannot read from webcam.")
    exit()

cam_h, cam_w = frame.shape[:2]

# Set initial transformation values
scale = 1.0
angle = 0
offset = np.array([(cam_w - img_w) // 2, (cam_h - img_h) // 2])  # Image centered

# Track initial gesture state
start_distance = None
start_angle = None
start_center = None

# Setup for smoothing movements (Exponential Moving Average)
SMOOTHING = 0.2
smooth_scale = scale
smooth_angle = angle
smooth_offset = offset.copy()

# Calculate Euclidean distance between two points
def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Calculate angle (in degrees) between two points
def calc_angle(p1, p2):
    delta = np.array(p2) - np.array(p1)
    return math.degrees(math.atan2(delta[1], delta[0]))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the frame for intuitive control
    cam_h, cam_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # If a hand is detected and it is the right hand
    if results.multi_hand_landmarks and results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label
        if handedness == "Right":
            hand = results.multi_hand_landmarks[0]

            # Track landmarks: Thumb (4), Index (8), Wrist (0)
            thumb = hand.landmark[4]
            index = hand.landmark[8]
            wrist = hand.landmark[0]

            # Convert normalized coordinates to pixel coordinates
            p_thumb = (int(thumb.x * cam_w), int(thumb.y * cam_h))
            p_index = (int(index.x * cam_w), int(index.y * cam_h))
            p_wrist = np.array([int(wrist.x * cam_w), int(wrist.y * cam_h)])

            # Current frame values
            center_now = p_wrist
            dist_now = calc_distance(p_thumb, p_index)
            angle_now = calc_angle(p_thumb, p_index)

            # Store initial frame values
            if start_distance is None:
                start_distance = dist_now
                start_angle = angle_now
                start_center = center_now
            else:
                # Calculate target transformations
                target_scale = dist_now / start_distance
                target_angle = angle_now - start_angle
                target_offset = center_now - start_center + np.array([(cam_w - img_w) // 2, (cam_h - img_h) // 2])

                # Apply smoothing
                smooth_scale = (1 - SMOOTHING) * smooth_scale + SMOOTHING * target_scale
                smooth_angle = (1 - SMOOTHING) * smooth_angle + SMOOTHING * target_angle
                smooth_offset = (1 - SMOOTHING) * smooth_offset + SMOOTHING * target_offset
        else:
            # Reset if left hand is shown
            start_distance = None
            start_angle = None
    else:
        # Reset if no hand is detected
        start_distance = None
        start_angle = None

    # Apply rotation and scaling to the image
    M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), smooth_angle, smooth_scale)
    transformed = cv2.warpAffine(image.copy(), M, (img_w, img_h))

    # Create blank canvas and calculate overlay positions
    canvas = np.zeros_like(frame)
    x_off, y_off = smooth_offset.astype(int)
    x1 = max(0, x_off)
    y1 = max(0, y_off)
    x2 = min(cam_w, x_off + img_w)
    y2 = min(cam_h, y_off + img_h)

    tx1 = max(0, -x_off)
    ty1 = max(0, -y_off)
    tx2 = tx1 + (x2 - x1)
    ty2 = ty1 + (y2 - y1)

    try:
        # Place the transformed image onto the canvas
        canvas[y1:y2, x1:x2] = transformed[ty1:ty2, tx1:tx2]
    except:
        pass  # Ignore edge-related exceptions

    # Draw hand landmarks on the original frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display original and canvas side by side
    stacked = np.hstack((frame, canvas))
    cv2.imshow("Right Hand Gesture Control | [Webcam | Transformed Image]", stacked)

    # Exit or reset
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('r'):
        scale = 1.0
        angle = 0
        offset = np.array([(cam_w - img_w) // 2, (cam_h - img_h) // 2])
        start_distance = None
        start_angle = None
        smooth_scale = scale
        smooth_angle = angle
        smooth_offset = offset.copy()
        print("🔁 Transformations reset")

# Release resources
cap.release()
cv2.destroyAllWindows()
