import cv2
import logging
import argparse
import warnings
import subprocess
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

from uniface import RetinaFace

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def extract_bbox(face):
    # Support both old/new uniface outputs (object with .bbox or dict with bbox/box key)
    if hasattr(face, "bbox"):
        return face.bbox

    if isinstance(face, dict):
        return face.get("bbox") or face.get("box")

    return None


def detect_screen_size(default_width, default_height):
    try:
        result = subprocess.run(
            ["xrandr"],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "*" not in line:
                continue
            for token in line.split():
                if "x" not in token:
                    continue
                if not token[0].isdigit():
                    continue
                width_str, height_str = token.split("x", 1)
                return int(width_str), int(height_str)
    except Exception:
        pass

    return default_width, default_height


def build_gaze_feature(yaw, pitch, bbox, frame_width, frame_height):
    x_min, y_min, x_max, y_max = map(float, bbox[:4])
    center_x = (x_min + x_max) * 0.5 / max(frame_width, 1)
    center_y = (y_min + y_max) * 0.5 / max(frame_height, 1)
    box_w = (x_max - x_min) / max(frame_width, 1)
    box_h = (y_max - y_min) / max(frame_height, 1)

    return np.array([yaw, pitch, center_x, center_y, box_w, box_h, 1.0], dtype=np.float32)


def generate_calibration_points(width, height, grid_size, margin_ratio=0.1):
    xs = np.linspace(margin_ratio * width, (1.0 - margin_ratio) * width, grid_size)
    ys = np.linspace(margin_ratio * height, (1.0 - margin_ratio) * height, grid_size)

    points = []
    for row, y in enumerate(ys):
        row_points = [(int(x), int(y)) for x in xs]
        if row % 2 == 1:
            row_points.reverse()
        points.extend(row_points)

    return points


class LinearCalibrationMapper:
    def __init__(self):
        self.features = []
        self.targets = []
        self.weights = None

    def add_sample(self, feature, target):
        self.features.append(feature.astype(np.float32))
        self.targets.append(np.array(target, dtype=np.float32))

    def fit(self):
        if len(self.features) < 6:
            raise ValueError("At least 6 calibration samples are required.")

        x = np.vstack(self.features)
        y = np.vstack(self.targets)
        self.weights, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    def is_ready(self):
        return self.weights is not None

    def predict(self, feature):
        if self.weights is None:
            return None
        return feature @ self.weights


class CalibrationSession:
    def __init__(self, points, hold_frames, move_frames=18, settle_frames=10, initial_wait_frames=75):
        self.points = points
        self.hold_frames = max(hold_frames, 1)
        self.move_frames = max(move_frames, 1)
        self.settle_frames = max(settle_frames, 0)
        self.initial_wait_frames = max(initial_wait_frames, 0)
        self.index = 0
        self.buffer = []
        self.phase = "initial_wait" if self.initial_wait_frames > 0 else "capture"
        self.phase_frame = 0

        start = np.array(self.points[0], dtype=np.float32)
        self.display_pos = start.copy()
        self.move_start = start.copy()

    def current_target(self):
        if self.index >= len(self.points):
            return None
        return self.points[self.index]

    def display_target(self):
        if self.index >= len(self.points):
            return None
        return int(self.display_pos[0]), int(self.display_pos[1])

    def status_text(self):
        if self.phase == "initial_wait":
            return "first point: focus and get ready"
        if self.phase == "move":
            return "follow the moving green dot"
        if self.phase == "settle":
            return "hold gaze, get ready"
        return "keep looking at the green dot"

    def _next_point(self):
        self.index += 1
        self.phase_frame = 0
        self.buffer.clear()

        if self.index >= len(self.points):
            return

        self.move_start = self.display_pos.copy()
        self.phase = "move"

    def update(self):
        target = self.current_target()
        if target is None:
            return True

        target_vec = np.array(target, dtype=np.float32)

        if self.phase == "initial_wait":
            self.display_pos = target_vec
            self.phase_frame += 1
            if self.phase_frame >= self.initial_wait_frames:
                self.phase = "capture"
                self.phase_frame = 0

        elif self.phase == "move":
            t = min(1.0, (self.phase_frame + 1) / float(self.move_frames))
            self.display_pos = (1.0 - t) * self.move_start + t * target_vec
            self.phase_frame += 1

            if t >= 1.0:
                self.display_pos = target_vec
                self.phase = "settle" if self.settle_frames > 0 else "capture"
                self.phase_frame = 0

        elif self.phase == "settle":
            self.display_pos = target_vec
            self.phase_frame += 1
            if self.phase_frame >= self.settle_frames:
                self.phase = "capture"
                self.phase_frame = 0

        else:
            self.display_pos = target_vec

        return False

    def add_feature(self, feature, mapper):
        target = self.current_target()
        if target is None:
            return False, True

        if self.phase != "capture":
            return False, False

        self.buffer.append(feature)
        if len(self.buffer) < self.hold_frames:
            return False, False

        averaged = np.mean(np.stack(self.buffer, axis=0), axis=0)
        mapper.add_sample(averaged, target)
        self._next_point()

        return True, self.index >= len(self.points)


def select_primary_face(faces):
    best_bbox = None
    best_area = -1.0

    for face in faces:
        bbox = extract_bbox(face)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = map(float, bbox[:4])
        area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
        if area > best_area:
            best_area = area
            best_bbox = bbox

    return best_bbox


def render_screen_view(
    width,
    height,
    calibrate,
    mapper_ready,
    calibration_target,
    calibration_idx,
    calibration_total,
    calibration_status,
    predicted_point,
    no_face,
):
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)

    if calibrate and not mapper_ready:
        if calibration_target is not None:
            cv2.circle(canvas, calibration_target, 18, (0, 220, 120), -1)
            cv2.putText(
                canvas,
                f"Calibration {calibration_idx + 1}/{calibration_total}: {calibration_status}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "Calibration complete, fitting mapper...",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            canvas,
            "Red dot = estimated gaze point",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if predicted_point is not None:
        cv2.circle(canvas, predicted_point, 14, (0, 0, 255), -1)

    if no_face:
        cv2.putText(
            canvas,
            "No face detected",
            (40, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 120, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze esimation model weights",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Display the inference results",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="assets/in_video.mp4",
        help="Path to source video file or camera index",
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaze360",
        help="Dataset name to get dataset related configs",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run on-screen calibration before gaze-to-screen mapping",
    )
    parser.add_argument(
        "--calib-grid",
        type=int,
        default=3,
        help="Calibration grid size (3 means 3x3 points)",
    )
    parser.add_argument(
        "--calib-hold",
        type=int,
        default=18,
        help="Number of frames to average per calibration point",
    )
    parser.add_argument(
        "--calib-move-frames",
        type=int,
        default=20,
        help="Frames for moving the calibration dot to the next point",
    )
    parser.add_argument(
        "--calib-settle-frames",
        type=int,
        default=10,
        help="Frames to wait at each point before sampling",
    )
    parser.add_argument(
        "--calib-initial-wait-frames",
        type=int,
        default=75,
        help="Frames to wait only for the first calibration point before sampling",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=0,
        help="Screen width override in pixels",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=0,
        help="Screen height override in pixels",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing factor for screen point, range (0, 1]",
    )
    parser.add_argument(
        "--fullscreen-screen",
        action="store_true",
        help="Show screen gaze view in fullscreen mode",
    )
    parser.add_argument(
        "--hide-screen",
        action="store_true",
        help="Disable gaze point screen window",
    )
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = RetinaFace()  # third-party face detection library

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")
        raise FileNotFoundError(f"Model weights not found at {params.weight}") from e

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.source
    is_webcam = video_source.isdigit()
    if is_webcam:
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter(params.output, fourcc, fps, (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    default_screen_w = frame_width
    default_screen_h = frame_height

    screen_w = params.screen_width
    screen_h = params.screen_height
    if screen_w <= 0 or screen_h <= 0:
        screen_w, screen_h = detect_screen_size(default_screen_w, default_screen_h)

    show_screen = not params.hide_screen
    if show_screen:
        cv2.namedWindow("Screen Gaze", cv2.WINDOW_NORMAL)
        if params.fullscreen_screen:
            cv2.setWindowProperty("Screen Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow("Screen Gaze", screen_w, screen_h)

    mapper = LinearCalibrationMapper()
    calibration_points = generate_calibration_points(screen_w, screen_h, max(params.calib_grid, 2))
    calibration = (
        CalibrationSession(
            calibration_points,
            params.calib_hold,
            move_frames=params.calib_move_frames,
            settle_frames=params.calib_settle_frames,
            initial_wait_frames=params.calib_initial_wait_frames,
        )
        if params.calibrate
        else None
    )
    smoothed_point = None

    if params.calibrate:
        logging.info("Calibration started. Please look at each green point until it changes.")

    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            if is_webcam:
                frame = cv2.flip(frame, 1)

            if calibration is not None and not mapper.is_ready():
                calibration.update()

            faces = face_detector.detect(frame) or []
            primary_bbox = select_primary_face(faces)

            current_feature = None
            no_face = primary_bbox is None
            if primary_bbox is not None:
                x_min, y_min, x_max, y_max = map(int, primary_bbox[:4])
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)

                image = frame[y_min:y_max, x_min:x_max]
                if image.size != 0:
                    image = pre_process(image)
                    image = image.to(device)

                    yaw_logits, pitch_logits = gaze_detector(image)
                    yaw_prob = F.softmax(yaw_logits, dim=1)
                    pitch_prob = F.softmax(pitch_logits, dim=1)

                    yaw_deg = torch.sum(yaw_prob * idx_tensor, dim=1) * params.binwidth - params.angle
                    pitch_deg = torch.sum(pitch_prob * idx_tensor, dim=1) * params.binwidth - params.angle

                    yaw_rad = float((yaw_deg * np.pi / 180.0).cpu().item())
                    pitch_rad = float((pitch_deg * np.pi / 180.0).cpu().item())

                    draw_bbox_gaze(frame, primary_bbox, pitch_rad, yaw_rad)
                    current_feature = build_gaze_feature(yaw_rad, pitch_rad, primary_bbox, frame_width, frame_height)

            if calibration is not None and current_feature is not None and not mapper.is_ready():
                advanced, done = calibration.add_feature(current_feature, mapper)
                if advanced:
                    logging.info(
                        f"Captured calibration point {calibration.index}/{len(calibration.points)}"
                    )
                if done:
                    mapper.fit()
                    calibration = None
                    logging.info("Calibration complete. Screen gaze mapping is now active.")

            predicted_point = None
            if current_feature is not None:
                mapped = mapper.predict(current_feature)
                if mapped is None:
                    # Fallback motion before calibration; not meant for accurate cursor control.
                    mapped = np.array(
                        [
                            screen_w * (0.5 - current_feature[0] / 1.2),
                            screen_h * (0.5 - current_feature[1] / 1.2),
                        ],
                        dtype=np.float32,
                    )

                mapped[0] = np.clip(mapped[0], 0, screen_w - 1)
                mapped[1] = np.clip(mapped[1], 0, screen_h - 1)

                alpha = float(np.clip(params.smooth_alpha, 0.01, 1.0))
                if smoothed_point is None:
                    smoothed_point = mapped
                else:
                    smoothed_point = alpha * mapped + (1.0 - alpha) * smoothed_point

                predicted_point = (int(smoothed_point[0]), int(smoothed_point[1]))

            if show_screen:
                calibration_target = calibration.display_target() if calibration is not None else None
                calibration_idx = calibration.index if calibration is not None else len(calibration_points)
                calibration_status = calibration.status_text() if calibration is not None else ""
                screen_canvas = render_screen_view(
                    width=screen_w,
                    height=screen_h,
                    calibrate=params.calibrate,
                    mapper_ready=mapper.is_ready() or not params.calibrate,
                    calibration_target=calibration_target,
                    calibration_idx=calibration_idx,
                    calibration_total=len(calibration_points),
                    calibration_status=calibration_status,
                    predicted_point=predicted_point,
                    no_face=no_face,
                )
                cv2.imshow("Screen Gaze", screen_canvas)

            if params.output:
                out.write(frame)

            if params.view:
                cv2.imshow("Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output and args.hide_screen:
        raise Exception("At least one of --view or --ouput must be provided.")

    main(args)
