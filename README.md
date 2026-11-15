# Real-Time Eye Detection and Cropping with MediaPipe

This project demonstrates real-time eye detection, cropping, and visualization using MediaPipe Face Mesh and OpenCV. It captures video from a webcam, identifies left and right eyes, draws bounding boxes on them, and displays cropped versions in separate windows.

## How it works

The crop boxes come from four eye landmarks of the MediaPipe Face Mesh (the eye corners and the upper and lower lid points) plus a 20 pixel margin. Iris refinement is switched on in the Face Mesh, but the code does not use the iris or pupil position for the crop. The repository name says pupil detection, and what the code does is eye detection and cropping.

## Features

1. Real-time face mesh detection using MediaPipe
2. Accurate left and right eye detection (corrected for webcam mirror flip)
3. Automatic eye cropping with bounding boxes
4. Cropped eyes displayed in separate resizable windows
5. Consistent crop size for both eyes (for clarity and ML preprocessing)
6. Simple exit mechanism (`q` key)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Three windows will appear:
- **Webcam with Eye Boxes** — full webcam feed with bounding boxes
- **Left Eye** — cropped and resized left eye
- **Right Eye** — cropped and resized right eye

Press `q` to exit.

Check `image_samples/` for sample output (images and video clips).

## Future Improvements

- Pupil center detection inside cropped eyes
- Blink detection and eye aspect ratio
- Eye gaze direction tracking
- Save cropped eye frames for machine learning tasks

## License

Licensed under the [MIT License](LICENSE).
