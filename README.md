# Human Action Recognition using CNN + LSTM (PyTorch)

## Overview

This project implements a **Human Action Recognition (HAR)** system using a hybrid deep learning model:

*  CNN (ResNet18) → Extract spatial features from frames
*  LSTM → Learn temporal patterns across frames

The system takes a video as input and predicts the human action being performed.

---

##  Features

* End-to-end video classification pipeline
* Frame extraction using OpenCV
* Temporal frame sampling
* Pretrained CNN backbone (ResNet18)
* LSTM for sequence modeling
* Confusion matrix visualization
* GPU support (CUDA)

---

##  Project Structure

```
human-action-recognition/
│
├── data/                    # Dataset (UCF101 format)
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── main.py                 # Training + evaluation script
├── README.md
```

---

## Dataset

Dataset used: **UCF101**

* 101 action classes
* Real-world videos (sports, daily activities)

Download:
https://www.crcv.ucf.edu/data/UCF101.php


---

## Installation

```

pip install torch torchvision numpy opencv-python matplotlib seaborn scikit-learn
```

---

## Model Architecture

### 1 CNN (Feature Extraction)

* Model: ResNet18 (pretrained)
* Output: 512-dimensional feature vector per frame

### 2 LSTM (Temporal Learning)

* Input size: 512
* Hidden size: 256
* Sequence length: 16 frames

### 3 Classifier

* Fully connected layer → action class prediction

---

##  Pipeline Flow

```
Video
  ↓
Frame Extraction (OpenCV)
  ↓
Frame Sampling (16 frames)
  ↓
CNN (ResNet18)
  ↓
LSTM
  ↓
Fully Connected Layer
  ↓
Prediction
```

---

## Training

Hyperparameters:

```
Epochs: 5
Batch Size: 4
Learning Rate: 0.001
```

Run:

```
python main.py
```

---

## Evaluation

* Accuracy calculation
* Confusion Matrix (Seaborn heatmap)

Example Output:

```
Epoch [1/2], Loss: 1.2345
Epoch [2/2], Loss: 0.9876
Accuracy: 78.45%
```

---

## Confusion Matrix

* Visualizes model predictions vs actual labels
* Helps identify misclassified actions

---

## Core Components

### Video Processing

* `extract_frames()` → Extract frames from video
* `sample_frames()` → Uniform frame selection
* `process_video()` → Full preprocessing pipeline

### Dataset Class

* Custom PyTorch Dataset
* Dynamically loads videos and labels

### DataLoader

```
create_dataloader(data_dir, batch_size)
```

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib & Seaborn
* Scikit-learn

---

## Limitations

* Uses limited classes (currently set to 10)
* No data augmentation
* No validation split
* Low training epochs

---

## Future Improvements

* Train on full UCF101 dataset
* Add data augmentation
* Use 3D CNNs (C3D / I3D)
* Add attention mechanism
* Real-time inference (webcam)

---

## Notes

* Ensure videos are readable (not corrupted)
* Empty videos return `None`
* GPU recommended for faster training

---

## Contributing

Feel free to fork and improve:

* Model performance
* Code structure
* Deployment

---


## Acknowledgements

* UCF101 Dataset
* PyTorch
* OpenCV

## Support

If you found this project useful, consider giving it a ⭐ on GitHub!
