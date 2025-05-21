import cv2
import mediapipe as mp
import numpy as np
import math
from sklearn.svm import SVC
from skimage.feature import hog


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

def scale_image(img, scale):
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return img
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if new_w > w or new_h > h:
        x_crop = (new_w - w) // 2
        y_crop = (new_h - h) // 2
        cropped = resized[y_crop:y_crop + h, x_crop:x_crop + w]
        return cropped
    else:
        canvas = np.zeros_like(img)
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

input_img = cv2.imread("image.jpeg")
if input_img is None:
    print("Error: Could not load image.jpeg")
    exit()

cap = cv2.VideoCapture(0)

svm = SVC(probability=True)

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8,8),
                              cells_per_block=(2,2), block_norm='L2-Hys',
                              visualize=True, feature_vector=True)
    return features


prev_distance = None
current_scale = 1.0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    blurred = cv2.GaussianBlur(frame, (5,5), 0)

    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        results = hands.process(rgb)

    h, w, _ = frame.shape
    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            dist = distance(index_tip, thumb_tip)
            x_min = min(int(lm[i].x * w) for i in range(21))
            y_min = min(int(lm[i].y * h) for i in range(21))
            x_max = max(int(lm[i].x * w) for i in range(21))
            y_max = max(int(lm[i].y * h) for i in range(21))
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                features = extract_hog_features(roi)
                if prev_distance is not None:
                    diff = dist - prev_distance
                    if diff > 10:
                        gesture = "Zoom In"
                        current_scale += 0.05
                    elif diff < -10:
                        gesture = "Zoom Out"
                        current_scale -= 0.03
                    current_scale = max(0.2, min(current_scale, 3.0))

            prev_distance = dist
    else:
        prev_distance = None

    transformed_img = scale_image(input_img, current_scale)
    transformed_img = cv2.medianBlur(transformed_img, 3)

    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Webcam Feed", frame)
    cv2.imshow("Zoomed Image", transformed_img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
