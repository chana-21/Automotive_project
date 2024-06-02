import cv2
import face_recognition
import numpy as np
import os


def collect_face_data(user_name):
    save_dir = 'face_dataset'
    cap = cv2.VideoCapture(0)
    count = 0

    user_dir = os.path.join(save_dir, user_name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in faces:
            face = frame[top:bottom, left:right]

            if face.size > 0:
                cv2.imwrite(os.path.join(user_dir, f'{user_name}_{count}.jpg'), frame)
                count += 1
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.imshow('Collecting Faces', frame)

        if count == 20:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    user_name = 'conan'
    collect_face_data(user_name)
