import cv2
import os
import dlib


def collect_face_data(user_name):
    save_dir = 'dataset'
    total_images = 300
    saved_images = 0

    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()

    cv2.namedWindow('Collecting Faces', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Collecting Faces', 1280, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    user_dir = os.path.join(save_dir, user_name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    while saved_images < total_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y:y + h, x:x + w]

            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                cv2.imwrite(os.path.join(user_dir, f'{user_name}_{saved_images:04d}.jpg'), frame)
                saved_images += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Collecting Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    user_name = 'conan'
    collect_face_data(user_name)
