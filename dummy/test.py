import face_recognition
import os
import cv2
import numpy as np

KNOWN_FACES_DIR = 'face_dataset'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
known_faces = []
known_names = []


def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value range is 97 to 122, subtract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


def load_face_data():
    print('Loading known faces...')
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            locations = face_recognition.face_locations(small_frame)
            encodings = face_recognition.face_encodings(small_frame, locations)

            face_names = []
            for face_encoding in encodings:
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, str(name), (left + 10, bottom + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), FONT_THICKNESS)


        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_face_data()
    recognize_faces()
