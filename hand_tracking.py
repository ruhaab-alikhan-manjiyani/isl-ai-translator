import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create hand landmarker
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

# Previous wrist position
prev_x = 0
prev_y = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    detection_result = detector.detect(mp_image)

    # Draw landmarks
    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            # Wrist landmark = landmark[0]
            wrist = hand_landmarks[0]

            current_x = wrist.x
            current_y = wrist.y

            # Movement difference
            dx = current_x - prev_x
            dy = current_y - prev_y

            # Detect direction
            if dx > 0.03:
                cv2.putText(frame, "Moving Right", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            elif dx < -0.03:
                cv2.putText(frame, "Moving Left", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

            if dy > 0.03:
                cv2.putText(frame, "Moving Down", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

            elif dy < -0.03:
                cv2.putText(frame, "Moving Up", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

            # Save current position
            prev_x = current_x
            prev_y = current_y

            # Draw all landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

    # Show webcam
    cv2.imshow("Hand Tracking", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()