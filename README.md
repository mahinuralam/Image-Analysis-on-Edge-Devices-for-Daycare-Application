# Image Analysis on Edge Devices for Daycare

## Overview
- Edge-ready computer vision project for daycare safety.
- Face identification: MTCNN detections + InceptionResnet embeddings, evaluated with Transformer and lightweight classifiers (~98% accuracy).
- Action recognition: MoveNet pose keypoints + BiLSTM classifier (~97% accuracy).

## Key Notebooks
- `FaceDetection&Recognition.ipynb` – end-to-end face pipeline (data prep, detection, embeddings, classifiers).
- `Pose_Estimation&Action_Recognition.ipynb` – pose extraction, sequence building, action model training, on-device inference.

## Datasets
- `Datasets/Korean child & Pins dataset/` – labeled portraits for identity recognition.
- `Datasets/Action dataset/` – GIFs, extracted frames, pose keypoints, and the trained `action_recognition_model.h5`.
  
Run notebooks from the project root so relative paths resolve correctly.

## Quick Start
1. Create a Python 3.10+ environment.
2. Install core dependencies:
	```bash
	pip install facenet-pytorch mtcnn mediapipe tensorflow torch torchvision opencv-python tqdm
	```
3. Open the notebooks in Jupyter or VS Code and run cells in order. GPU support is optional but speeds up embedding extraction and model training.

## Workflow Summary
- Mirror or regenerate processed data using the setup cells (`processed_split`, `train_detected_faces`, pose keypoints).
- Train/evaluate face models (centroid matching, MLP, Transformer) on saved embeddings.
- Train/evaluate action models on sliding-window pose sequences; export to `.h5` for deployment.
- Optional cells demonstrate on-device inference and visualization.

## Results
- Face recognition accuracy: ~98% (Transformer classifier).
- Action recognition accuracy: ~97% (BiLSTM on MoveNet keypoints).

