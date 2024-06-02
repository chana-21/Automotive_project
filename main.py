import face_recognition
import os
import cv2
import numpy as np

KNOWN_FACES_DIR = 'face_dataset'
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
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
    cap = cv2.VideoCapture(0)
    process_this_frame = True
    face_names = []
    locations = []
    encodings = []

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cv2.resizeWindow('Video', width * 2, height * 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if process_this_frame:
            # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = frame

            locations = face_recognition.face_locations(small_frame)
            encodings = face_recognition.face_encodings(small_frame, locations)

            face_names = []
            for face_encoding in encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "unknown"

                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(locations, face_names):
            # top *= 4
            # right *= 4
            # bottom *= 4
            # left *= 4

            if name == "unknown":
                face_region = frame[top:bottom, left:right]
                face_region_blurred = cv2.GaussianBlur(face_region, (99, 99), 30)

                mask = np.zeros(face_region.shape[:2], dtype=np.uint8)
                mask_center = (mask.shape[1] // 2, mask.shape[0] // 2)
                mask_radius = min(mask_center[0], mask_center[1], (right - left) // 2, (bottom - top) // 2)
                cv2.circle(mask, mask_center, mask_radius, 255, -1)

                mask_inv = cv2.bitwise_not(mask)
                face_region = cv2.bitwise_and(face_region, face_region, mask=mask_inv)
                face_region = cv2.add(face_region, cv2.bitwise_and(face_region_blurred, face_region_blurred, mask=mask))

                frame[top:bottom, left:right] = face_region
            # else:
            #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), FRAME_THICKNESS)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_face_data()
    recognize_faces()
