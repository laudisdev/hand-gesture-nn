
import time
import math
import torch

import cv2
import mediapipe as mp

from model import GestureNet



def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Unable to open your camera.")

        return

    base_options = mp.tasks.BaseOptions(model_asset_path="models/hand_landmarker.task")
    running_mode = mp.tasks.vision.RunningMode.VIDEO
    
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options, 
        running_mode=running_mode, 
        num_hands=1
    )

    detector = mp.tasks.vision.HandLandmarker.create_from_options(options=options)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),                     # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),                     # Index
        (5, 9), (9, 10), (10, 11), (11, 12),                # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),              # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)     # Pinky
    ]

    gesture_names = ["Open Palm", "Fist", "Peace"]

    model = GestureNet()
    model.eval()
    
    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.monotonic() * 1000)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            height, width = frame.shape[:2]

            wrist = hand[0]
            base_middle_finger = hand[9]

            dx = base_middle_finger.x - wrist.x
            dy = base_middle_finger.y - wrist.y

            hand_size = math.hypot(dx, dy)
            hand_size = max(hand_size, 1e-6)

            points = []

            features = []

            for landmark in hand:
                pixel_x = int(landmark.x * width)
                pixel_y = int(landmark.y * height)
                points.append((pixel_x, pixel_y))
                relative_x = (landmark.x - wrist.x) / hand_size
                relative_y = (landmark.y - wrist.y) / hand_size
                relative_z = (landmark.z - wrist.z) / hand_size

                features.extend([relative_x, relative_y, relative_z])

                cv2.circle(frame, (pixel_x, pixel_y), 6, (0, 255, 0), -1)

            hand_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(dim=0)

            with torch.no_grad():
                output = model(hand_tensor)
                probabilities = torch.softmax(output, dim=1)
                print(probabilities)
                predicted_class = torch.argmax(output, dim=1)

                class_index = int(predicted_class.item())

                gesture_name = gesture_names[class_index]

                cv2.putText(frame, gesture_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for start, end in connections:
                cv2.line(frame, points[start], points[end], (255, 0, 0), 2) 
            
        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
