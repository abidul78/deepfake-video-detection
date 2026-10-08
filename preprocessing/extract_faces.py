import cv2
import os


def extract_faces(video_path, output_folder, frame_interval=10):

    os.makedirs(output_folder, exist_ok=True)

    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = 0
    processed_frames = 0
    saved_faces = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        # Process every 10th frame
        if total_frames % frame_interval == 0:

            processed_frames += 1

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # If multiple faces exist, use the largest face
            if len(faces) > 0:

                x, y, w, h = max(
                    faces,
                    key=lambda face: face[2] * face[3]
                )

                # Crop face from original colour frame
                face = frame[y:y+h, x:x+w]

                # Resize for CNN input
                face = cv2.resize(face, (224, 224))

                face_path = os.path.join(
                    output_folder,
                    f"face_{saved_faces:04d}.jpg"
                )

                cv2.imwrite(face_path, face)

                saved_faces += 1

        total_frames += 1

    video.release()

    print("Face extraction completed.")
    print("Total video frames:", total_frames)
    print("Frames checked:", processed_frames)
    print("Faces saved:", saved_faces)


if __name__ == "__main__":

    extract_faces(
        "test_video.mp4",
        "test_faces",
        frame_interval=10
    )