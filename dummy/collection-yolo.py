import cv2
import os


# 웹캠을 사용해 얼굴 데이터 수집
def collect_face_data(user_name, save_dir='face_dataset'):
    cap = cv2.VideoCapture(0)
    count = 0

    user_dir = os.path.join(save_dir, user_name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 검출
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(user_dir, f'{user_name}_{count}.jpg'), face)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색 박스로 표시

        cv2.imshow('Collecting Faces', frame)

        if count == 200:
            break

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_face_data('chana')
