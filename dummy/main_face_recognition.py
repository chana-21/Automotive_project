import cv2
import numpy as np
import os
import face_recognition

# 저장된 얼굴 데이터 불러오기
data_dir = 'face_data'
known_faces = np.load(os.path.join(data_dir, 'known_faces.npy'))

# 웹캠 초기화
video_capture = cv2.VideoCapture(0)

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
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # 녹색 박스
        else:
            # 얼굴 영역에 블러 처리 (동그라미 형태)
            face_region = frame[top:bottom, left:right]
            blurred_face = cv2.GaussianBlur(face_region, (21, 21), 5)  # 약한 블러

            # 동그라미 마스크 생성
            mask_circle = np.zeros_like(face_region, dtype=np.uint8)
            center = (right - left) // 2, (bottom - top) // 2
            radius = min(right - left, bottom - top) // 2
            cv2.circle(mask_circle, center, radius, (255, 255, 255), -1, cv2.LINE_AA)

            # 마스크를 사용하여 동그라미 모양으로 블러 적용
            masked_face = np.where(mask_circle == 255, blurred_face, face_region)
            frame[top:bottom, left:right] = masked_face

    # 결과 프레임을 표시합니다
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
