import cv2
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# LOAD AI MODEL
# =========================
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# LOAD MEDIAPIPE MODEL
# =========================
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

# Create detector
landmarker = vision.HandLandmarker.create_from_options(options)

# =========================
# OPEN WEBCAM
# =========================
cap = cv2.VideoCapture(0)

# Widescreen resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Resizable window
cv2.namedWindow("ISL AI Translator", cv2.WINDOW_NORMAL)

print("ISL AI Translator Running...")
print("Press Q to Quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # =========================
    # DARK OVERLAY
    # =========================
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (1280, 120),
        (15, 15, 15),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    # =========================
    # CONVERT TO RGB
    # =========================
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # =========================
    # HAND DETECTION
    # =========================
    detection_result = landmarker.detect(mp_image)

    prediction = "No Gesture"

    hand_count = 0

    if detection_result.hand_landmarks:

        hand_count = len(detection_result.hand_landmarks)

        for hand_landmarks in detection_result.hand_landmarks:

            row = []

            for landmark in hand_landmarks:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Predict gesture
            prediction = model.predict([row])[0]

            # Draw landmark points
            for landmark in hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 255, 255),
                    -1
                )

    # =========================
    # HUD TEXT
    # =========================

    # Title
    cv2.putText(
        frame,
        "ISL AI TRANSLATOR | BY RUHAAB",
        (30, 45),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Gesture prediction
    cv2.putText(
        frame,
        f"Gesture: {prediction}",
        (30, 85),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (220, 220, 220),
        2
    )

    # Hand count
    cv2.putText(
        frame,
        f"Hands Detected: {hand_count}",
        (950, 45),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (180, 180, 180),
        2
    )

    # Exit hint
    cv2.putText(
        frame,
        "Press Q to Exit",
        (950, 85),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (180, 180, 180),
        2
    )

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow("ISL AI Translator", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()