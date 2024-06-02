import cv2
import numpy as np
import os
import face_recognition

# 얼굴 데이터 저장 경로
data_dir = 'face_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

video_capture = cv2.VideoCapture(0)

collected_face_encodings = []

print("Press 'c' to capture and 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # 얼굴 위치 감지
    face_locations = face_recognition.face_locations(frame)

    # 얼굴 인코딩
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        collected_face_encodings.append(face_encoding)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)  # 파란색 박스로 표시

    # 결과 프레임을 표시합니다
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if len(face_encodings) > 0:
            print("Faces captured and added.")
    elif key == ord('q'):
        break

# 얼굴 데이터 저장
face_encodings_array = np.array(collected_face_encodings)
np.save(os.path.join(data_dir, 'known_faces.npy'), face_encodings_array)

video_capture.release()
cv2.destroyAllWindows()
