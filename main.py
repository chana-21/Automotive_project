import cv2

# Load the pre-trained Haar Cascade classifier for face detection using CUDA
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame to grayscale as the face detector expects gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Upload the grayscale image to the GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(gray)

    # Detect faces in the frame using CUDA
    faces = face_cascade.detectMultiScale(gpu_frame)

    # Download the results back to the CPU
    faces = faces.download()

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
