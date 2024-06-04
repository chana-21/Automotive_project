import cv2
import os
import dlib
import splitfolders
import shutil


def get_dir_path(base_path, temp_path, name):
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    count = 0
    target_dir = None

    for d in existing_dirs:
        try:
            num, folder_name = d.split('-', 1)
            num = int(num)
            if folder_name == name:
                target_dir = os.path.join(base_path, d)
                break
            if num > count:
                count = num
        except ValueError:
            continue

    if target_dir is None:
        new_folder_name = f"{count + 1}-{name}"
        target_dir = os.path.join(temp_path, new_folder_name)
        os.makedirs(target_dir, exist_ok=True)

    return target_dir, count + 1


def collect_face_data(output_dir):
    # 얼굴 탐지기 초기화
    detector = dlib.get_frontal_face_detector()

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)

    total_images = 300

    # 저장한 이미지 수 초기화
    saved_images = 0

    while saved_images < total_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y:y+h, x:x+w]

            # 얼굴 크롭 이미지의 크기가 유효한지 확인
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                save_path = os.path.join(output_dir, f'face_{saved_images:04d}.jpg')
                cv2.imwrite(save_path, face_img)
                saved_images += 1

                # 얼굴 영역에 바운딩 박스 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if saved_images >= total_images:
                    break

        cv2.imshow('face data collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def generate_labels(dataset_image_path):
    dataset_label_path = dataset_image_path.replace('image', 'labels')

    for split in ['train', 'val']:
        split_image_path = os.path.join(dataset_image_path, split)
        split_label_path = os.path.join(dataset_label_path, split)

        class_dirs = [d for d in os.listdir(split_image_path) if os.path.isdir(os.path.join(split_image_path, d))]

        for class_dir in class_dirs:
            class_id = int(class_dir.split('-')[0])
            image_dir = os.path.join(split_image_path, class_dir)
            label_dir = os.path.join(split_label_path, class_dir)

            os.makedirs(label_dir, exist_ok=True)

            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

            for image_file in image_files:
                image_path = os.path.join(image_dir, image_file)
                label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))

                x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0  # 크롭된 이미지이므로 중심이 0.5, 크기가 1
                label_content = f"{class_id} {x_center} {y_center} {width} {height}\n"

                with open(label_path, 'w') as label_file:
                    label_file.write(label_content)


def move_files(dataset_path):
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_path, split)
        image_dir = os.path.join(split_dir, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            if os.path.isdir(class_path):
                dest_dir = os.path.join(image_dir, class_dir)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                for file_name in os.listdir(class_path):
                    src_file = os.path.join(class_path, file_name)
                    dest_file = os.path.join(dest_dir, file_name)
                    shutil.move(src_file, dest_file)
                os.rmdir(class_path)


if __name__ == '__main__':
    user_name = 'chana'

    dataset_path = 'dataset-test'

    temp_path = os.path.join(dataset_path, 'temp')
    os.makedirs(temp_path, exist_ok=True)

    target_path, class_id = get_dir_path(dataset_path, temp_path, user_name)

    # collect_face_data(target_path)

    splitfolders.ratio(temp_path, output=dataset_path, seed=1337, ratio=(.8, .2))

    shutil.rmtree(temp_path)

    move_files(dataset_path)

    # generate_labels(dataset_path)
