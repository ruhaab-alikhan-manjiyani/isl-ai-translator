import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
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

# Ask for gesture label
label = input("Enter gesture label: ")

# Open CSV file
file = open("gesture_data.csv", mode="a", newline="")
writer = csv.writer(file)

print("Collecting data... Press Q to stop.")

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            row = []

            for landmark in hand_landmarks:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Add label
            row.append(label)

            # Save row
            writer.writerow(row)

            # Draw points
            for landmark in hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()