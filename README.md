# Image Classifier using ResNet-18

## Project Overview

This project implements an end-to-end image classification pipeline using a pretrained ResNet-18 model and the CIFAR-10 dataset. Transfer learning and data augmentation techniques are used to improve model performance. The project supports model training, evaluation, checkpoint saving, and inference on custom images.

---

## Dataset

**Dataset:** CIFAR-10

* Training Images: 50,000
* Testing Images: 10,000
* Classes: 10
* Image Size: 32 × 32 RGB

### Classes

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## Features

* Data Loading using PyTorch
* Data Augmentation
* Transfer Learning with ResNet-18
* Model Evaluation
* Best Model Checkpoint Saving
* Custom Image Inference
* GitHub Ready Project Structure

---

## Model

**Architecture:** ResNet-18

**Transfer Learning:** Yes

**Framework:** PyTorch

---

## Training Configuration

* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Batch Size: 64
* Epochs: 10
* Learning Rate: 0.001

---

## Results

| Metric             |     Value |
| ------------------ | --------: |
| Best Test Accuracy |    83.28% |
| Epochs             |        10 |
| Model              | ResNet-18 |

---

## Project Structure

```text
image_classifier_project/
│
├── main.py
├── inference.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── best_model.pth
│
├── test_images/
│
└── data/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python main.py
```

---

## Inference

Place an image inside the `test_images` folder and run:

```bash
python inference.py test_images/image.jpg
```

Example Output:

```text
Predicted Class: dog
```

---

## Conclusion

The pretrained ResNet-18 model achieved a best test accuracy of **83.28%** on the CIFAR-10 dataset. Transfer learning and data augmentation significantly improved image classification performance while enabling efficient training and inference on new images.

---

## Author

Kapil Kumawat

B.Tech CSE (Artificial Intelligence)

JK Lakshmipat University, Jaipur
