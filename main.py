import dlib
import face_recognition
import os
import cv2
import numpy as np

KNOWN_FACES_DIR = 'dataset'
TOLERANCE = 0.6
known_faces = []
known_names = []


def load_face_data():
    print("Loading known faces...")
    for name in os.listdir(KNOWN_FACES_DIR):
        print(name)
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)


def recognize_faces():
    cap = cv2.VideoCapture('Conan_show.mp4')
    # cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 1280, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'H264' if 'mp4v' doesn't work
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        locations = [(face.top(), face.right(), face.bottom(), face.left()) for face in faces]
        encodings = face_recognition.face_encodings(frame, locations)

        face_names = []
        for face_encoding in encodings:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "unknown"

            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(locations, face_names):

            if name == "unknown":
                face_region = frame[top:bottom, left:right]

                if face_region.size == 0:
                    continue

                face_region_blurred = cv2.GaussianBlur(face_region, (99, 99), 30)

                mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
                mask_center = (mask.shape[1] // 2, mask.shape[0] // 2)
                mask_radius = int(min(mask_center[0], mask_center[1], (right - left) // 2, (bottom - top) // 2))
                cv2.circle(mask, mask_center, mask_radius, 255, -1)

                mask_inv = cv2.bitwise_not(mask)
                face_region = cv2.bitwise_and(face_region, face_region, mask=mask_inv)
                face_region = cv2.add(face_region, cv2.bitwise_and(face_region_blurred, face_region_blurred, mask=mask))

                frame[top:bottom, left:right] = face_region
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        out.write(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_face_data()
    recognize_faces()
