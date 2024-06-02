import cv2
import numpy as np
import os

# Haar Cascade 파일 경로
cascPathProfile = "data/haarcascade_profileface.xml"

# 얼굴 인식용 Haar Cascade 생성
faceCascadeProfile = cv2.CascadeClassifier(cascPathProfile)

# 저장된 얼굴 데이터 불러오기
data_dir = 'face_data'
known_faces = np.load(os.path.join(data_dir, 'known_faces.npy'))

# 고정된 얼굴 이미지 크기
fixed_face_size = (100, 100)

# 웹캠 초기화
video_capture = cv2.VideoCapture(0)

# 임계값 설정
threshold = 90  # 이 값을 조정하여 정확도를 높일 수 있습니다.

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
    mask = np.zeros_like(gray)  # 처리된 영역을 기록할 마스크 배열

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
    skip_indices = set()
    for i, face in enumerate(all_faces):
        if i in skip_indices:
            continue
        for j, other_face in enumerate(all_faces):
            if i != j and j not in skip_indices:
                iou = calculate_iou(face[:4], other_face[:4])
                if iou > 0.3:  # 60%로 변경
                    skip_indices.add(j)
        filtered_faces.append(face)

    # 얼굴 처리
    for (x, y, w, h, is_flipped) in filtered_faces:
        if is_flipped:
            roi_gray = gray_flipped[y:y + h, x:x + w]
        else:
            roi_gray = gray[y:y + h, x:x + w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        resized_face = cv2.resize(roi_gray, fixed_face_size)
        min_diff = np.inf

        for known_face in known_faces:
            diff = np.mean(np.abs(known_face - resized_face))
            if diff < min_diff:
                min_diff = diff

        if min_diff >= threshold and np.sum(mask[y:y + h, x:x + w]) == 0:  # 아직 처리되지 않은 영역인지 확인
            # 얼굴 영역에 블러 처리 (동그라미 형태)
            face_region = frame[y:y + h, x:x + w]
            blurred_face = cv2.GaussianBlur(face_region, (21, 21), 5)  # 약한 블러

            # 동그라미 마스크 생성
            mask_circle = np.zeros_like(face_region, dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = min(w, h) // 2
            cv2.circle(mask_circle, center, radius, (255, 255, 255), -1, cv2.LINE_AA)

            # 마스크를 사용하여 동그라미 모양으로 블러 적용
            masked_face = np.where(mask_circle == 255, blurred_face, face_region)
            frame[y:y + h, x:x + w] = masked_face

            # 처리된 영역을 마스크에 기록
            mask[y:y + h, x:x + w] = 255

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
