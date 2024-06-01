import cv2
import numpy as np
import os

# Haar Cascade 파일 경로
cascPathProfile = "data/haarcascade_profileface.xml"

# 얼굴 인식용 Haar Cascade 생성
faceCascadeProfile = cv2.CascadeClassifier(cascPathProfile)

# 얼굴 데이터 저장 경로
data_dir = 'face_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

video_capture = cv2.VideoCapture(0)

collected_face_encodings = []
fixed_face_size = (100, 100)  # 고정된 얼굴 이미지 크기

print("Press 'c' to capture and 'q' to quit.")

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 측면 얼굴 검출 (원래 이미지)
    facesProfile = faceCascadeProfile.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 측면 얼굴 검출 (좌우 반전 이미지)
    gray_flipped = cv2.flip(gray, 1)
    facesProfileFlipped = faceCascadeProfile.detectMultiScale(
        gray_flipped,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 검출된 바운딩 박스를 하나의 리스트에 추가
    all_faces = [(x, y, w, h, False) for (x, y, w, h) in facesProfile]  # 원래 이미지의 측면 얼굴
    all_faces += [(gray.shape[1] - x - w, y, w, h, True) for (x, y, w, h) in facesProfileFlipped]  # 반전된 측면 얼굴

    # 겹치는 바운딩 박스 제거
    filtered_faces = []
    for i, face in enumerate(all_faces):
        keep = True
        for j, other_face in enumerate(all_faces):
            if i != j:
                iou = calculate_iou(face[:4], other_face[:4])
                if iou > 0.3:  # 60%로 변경
                    keep = False
                    break
        if keep:
            filtered_faces.append(face)

    # 검출된 측면 얼굴 처리
    for (x, y, w, h, is_flipped) in filtered_faces:
        if is_flipped:
            roi_gray = gray_flipped[y:y + h, x:x + w]
        else:
            roi_gray = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(roi_gray, fixed_face_size)
        collected_face_encodings.append(resized_face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색 박스로 표시

    # 결과 프레임을 표시합니다
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if len(filtered_faces) > 0:
            print("Faces captured and added.")
    elif key == ord('q'):
        break

# 얼굴 데이터 저장
face_encodings_array = np.array(collected_face_encodings)
np.save(os.path.join(data_dir, 'known_faces.npy'), face_encodings_array)

video_capture.release()
cv2.destroyAllWindows()