import os
import yaml
import subprocess

def create_data_yaml(image_path, output_path):
    # train 경로의 디렉토리 이름을 불러와 names 리스트를 만듦
    train_image_path = os.path.join(image_path, 'train')
    class_names = [d for d in os.listdir(train_image_path) if os.path.isdir(os.path.join(train_image_path, d))]

    # class_names를 이용해 class_id 순서를 정렬
    class_names.sort(key=lambda x: int(x.split('-')[0]))
    print(class_names)

    # class_names에서 '-' 이후 부분만 추출하여 최종 names 리스트 생성
    # names = [name.split('-')[1] for name in class_names]

    label_path = image_path.replace('image', 'labels')

    data_yaml = {
        'train': os.path.join(label_path, 'train'),
        'val': os.path.join(label_path, 'val'),
        'nc': len(class_names),  # 클래스 수
        'names': class_names
    }

    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f)


if __name__ == '__main__':
    dataset_path = os.path.join('dataset', 'labels')
    data_yaml_path = 'data.yaml'

    # data.yaml 파일 생성
    # create_data_yaml(dataset_path, data_yaml_path)

    # YOLOv7 저장소 클론 및 필요 패키지 설치
    if not os.path.exists('yolov7'):
        subprocess.run(['git', 'clone', 'https://github.com/WongKinYiu/yolov7'])

    # YOLOv7 학습
    subprocess.run([
        'python', os.path.join('yolov7', 'train.py'),
        '--img', '640',
        '--batch', '16',
        '--epochs', '30',
        '--data', 'data.yaml',
        '--cfg', 'cfg/training/yolov7.yaml',
        '--weights', os.path.join('yolov7', 'yolov7.pt'),
        '--device', 'cpu'
    ])
