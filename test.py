import cv2
import numpy as np

# Numpy 배열 생성
image = np.zeros((512, 512, 3), dtype=np.uint8)

# Numpy 배열을 GPU 메모리로 복사
d_image = cv2.cuda_GpuMat()
d_image.upload(image)

gray_image = cv2.cuda.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
result = gray_image.download()

cv2.imshow('gray_image', result)