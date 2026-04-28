import cv2
import mediapipe as mp
import csv
import os

os.makedirs("data/images", exist_ok=True)
DATABASE_FOLDER = "data/images"
FILE_NAME = 'data/hand_data.csv'


LABEL = 0  # 0: open, 1: closed, 2: closed_thumb, 3: pinch, 4: almost_pinch, 5: trash, 6: point
START_INDEX = 669


def collect_data():

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Collecting data for Label: {LABEL}")

    image_index = START_INDEX
    nb_images = 1

    while cap.isOpened():
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        image_to_save = image.copy()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data_row = []

                data_row.append(image_index)
                data_row.append(LABEL)

                for lm in hand_landmarks.landmark:
                    data_row.extend([lm.x, lm.y, lm.z])

                key = cv2.waitKey(1)
                if key == ord('s'):
                    with open(FILE_NAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data_row)
                    print(f"Image sauvegardée. Label n°{LABEL}, index n°{image_index}, nombre d'images = {nb_images}")
                    cv2.imwrite(f"data/images/{image_index}.png", image_to_save)

                    image_index +=1
                    nb_images +=1


                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow('Data Collector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if not os.path.exists(FILE_NAME):
        with open(FILE_NAME, 'a', newline='') as f:
            header =['image_index'] + ['label'] +  [f'pt{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]
            csv.writer(f).writerow(header)

    collect_data()