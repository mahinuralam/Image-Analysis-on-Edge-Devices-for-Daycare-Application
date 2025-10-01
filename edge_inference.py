import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

ACTIONS: List[str] = [
    "kick",
    "punch",
    "squat",
    "stand",
    "wave",
    "running",
    "sit",
    "fall",
]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Datasets" / "Action dataset"
DEFAULT_MODEL_PATH = DEFAULT_DATASET_ROOT / "edge_artifacts" / "action_recognition_model_float32.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TensorFlow Lite inference on pose keypoint sequences."
    )
    parser.add_argument(
        "--tflite-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the TFLite model (float32 or float16).",
    )
    parser.add_argument(
        "--keypoints-json",
        type=Path,
        required=True,
        help="Path to a JSON file containing keypoints (from pose_keypoints).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Number of frames per sequence window.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        help="Stride between consecutive windows.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display.",
    )
    return parser.parse_args()


def load_keypoints(json_path: Path) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    keypoints = payload["keypoints"]
    frames = np.array([np.array(frame, dtype=np.float32).flatten() for frame in keypoints])
    return frames


def create_subsequences(sequence: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    if sequence.shape[0] < window_size:
        pad_frames = np.zeros((window_size - sequence.shape[0], sequence.shape[1]), dtype=sequence.dtype)
        sequence = np.concatenate([sequence, pad_frames], axis=0)

    subsequences: List[np.ndarray] = []
    for start in range(0, sequence.shape[0] - window_size + 1, step_size):
        end = start + window_size
        subsequences.append(sequence[start:end])

    if not subsequences:
        subsequences.append(sequence[:window_size])

    return np.stack(subsequences)


def run_inference(model_path: Path, windows: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_index = input_details["index"]
    output_index = output_details["index"]
    input_dtype = input_details["dtype"]

    predictions: List[np.ndarray] = []
    for window in windows:
        input_tensor = np.expand_dims(window, axis=0).astype(input_dtype)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_index)[0])

    return np.array(predictions)


def summarise_predictions(predictions: np.ndarray, top_k: int) -> None:
    mean_probs = predictions.mean(axis=0)
    ranked_indices = mean_probs.argsort()[::-1][:top_k]

    print("Aggregated action probabilities:")
    for idx in ranked_indices:
        print(f"  {ACTIONS[idx]:>7}: {mean_probs[idx]:.3f}")

    best_idx = ranked_indices[0]
    print(f"\nPredicted action: {ACTIONS[best_idx]} (confidence {mean_probs[best_idx]:.3f})")



def main() -> None:
    args = parse_args()

    if not args.tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found at {args.tflite_path}")
    if not args.keypoints_json.exists():
        raise FileNotFoundError(f"Keypoint JSON not found at {args.keypoints_json}")

    sequence = load_keypoints(args.keypoints_json)
    windows = create_subsequences(sequence, args.window_size, args.step_size)

    print(f"Loaded {sequence.shape[0]} frames -> {windows.shape[0]} window(s) of shape {windows.shape[1:]}.")
    predictions = run_inference(args.tflite_path, windows)
    summarise_predictions(predictions, args.top_k)


if __name__ == "__main__":
    main()
