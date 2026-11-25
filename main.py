"""Real-time eye detection and cropping with MediaPipe's FaceLandmarker and OpenCV.

python main.py                  webcam 0, press q to quit
python main.py --camera 1       use another camera
python main.py --image face.jpg run once on a photo (use --no-show to skip the windows)
"""
import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Target size for cropped eye images
EYE_DISPLAY_SIZE = (300, 150)
MARGIN = 20  # pixels around the four eye landmarks

# Four landmarks per eye: the two corners, then the upper and lower lid points.
# Names follow the user's own eyes, so they are the opposite of MediaPipe's names.
LEFT_EYE_INDICES = [362, 263, 386, 374]   # MediaPipe's right eye
RIGHT_EYE_INDICES = [33, 133, 159, 145]   # MediaPipe's left eye


def ensure_model(path=MODEL_PATH):
    """Download the face landmarker model (about 4 MB) the first time it is needed."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_landmarker(running_mode):
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        running_mode=running_mode,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def eye_box(landmarks, indices, w, h):
    """Pixel box around four eye landmarks plus a margin, clipped to the image."""
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min = max(min(p[0] for p in points) - MARGIN, 0)
    y_min = max(min(p[1] for p in points) - MARGIN, 0)
    x_max = min(max(p[0] for p in points) + MARGIN, w)
    y_max = min(max(p[1] for p in points) + MARGIN, h)
    return x_min, y_min, x_max, y_max


def process_frame(frame, result):
    """Draw the eye boxes on frame, in place. Return (left_eye_crop, right_eye_crop) or Nones."""
    if not result.face_landmarks:
        return None, None
    h, w = frame.shape[:2]
    landmarks = result.face_landmarks[0]
    crops = []
    for indices, colour in ((LEFT_EYE_INDICES, (0, 0, 255)), (RIGHT_EYE_INDICES, (0, 255, 0))):
        x0, y0, x1, y1 = eye_box(landmarks, indices, w, h)
        crops.append(frame[y0:y1, x0:x1].copy())
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)  # red box left eye, green box right eye
    return crops[0], crops[1]


def show_crops(frame, left, right):
    for name, crop in (("Left Eye", left), ("Right Eye", right)):
        if crop is not None and crop.size > 0:
            cv2.imshow(name, cv2.resize(crop, EYE_DISPLAY_SIZE, interpolation=cv2.INTER_CUBIC))
    cv2.imshow("Webcam with Eye Boxes", cv2.resize(frame, (1280, 720)))


def run_image(path, show):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with make_landmarker(vision.RunningMode.IMAGE) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    left, right = process_frame(frame, result)
    if left is None:
        raise SystemExit("No face found in image.")
    print(f"Left eye crop {left.shape[1]}x{left.shape[0]}, right eye crop {right.shape[1]}x{right.shape[0]} pixels.")
    if show:
        show_crops(frame, left, right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(camera):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera}. Try --camera 0 or --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Allow resizable display windows
    for name in ("Webcam with Eye Boxes", "Right Eye", "Left Eye"):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    with make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)  # flip horizontally for a natural webcam feel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time() * 1000))
            left, right = process_frame(frame, result)
            show_crops(frame, left, right)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--image", help="run once on this image instead of the webcam")
    parser.add_argument("--no-show", action="store_true", help="with --image: do not open windows")
    args = parser.parse_args()
    print(f"MediaPipe {mp.__version__}, OpenCV {cv2.__version__}")
    if args.image:
        run_image(args.image, not args.no_show)
    else:
        run_webcam(args.camera)


if __name__ == "__main__":
    main()
