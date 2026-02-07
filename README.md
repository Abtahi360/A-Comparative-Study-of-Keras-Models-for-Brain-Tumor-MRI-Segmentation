A hands-on comparison of one model from each major Keras application family (35 models total in the original sweep — here we selected one representative from each family) trained and tested on a Brain Tumor MRI dataset. The goal: see how each pretrained Keras model behaves for this medical imaging task, compare qualitative outputs (segmentation / single predictions) and quantitative metrics (accuracy / loss / other relevant scores), and identify which models give the best trade-off between performance and cost.

---

## Table of contents (notebook flow)

1. Brain Tumor MRI Segmentation (dataset & preprocessing)
2. Xception
3. VGG19
4. ResNet50V2 *(notebook referenced “ResNetSOV2” — interpreted as ResNet50V2)*
5. Helper function for model creation
6. ResNet101V2
7. ResNet152V2
8. InceptionV3
9. InceptionResNetV2
10. MobileNetV2
11. DenseNet201
12. NASNetLarge
13. EfficientNetB7 *(interpreted from “EfficientNetg7” as EfficientNet-B7)*
14. EfficientNetV2-B3
15. EfficientNetV2-L
16. ConvNeXt-XLarge
17. Single prediction (inference demo)

---

## Project structure 

* `Types-of-keras-models.ipynb` - main notebook: loads data, preprocesses images, defines helper functions, instantiates models, trains/fine-tunes each selected architecture, evaluates, and shows single-image predictions and visual comparisons.

---

## Dataset

* Dataset: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
* Kaggle training location: [https://www.kaggle.com/code/abtahiislam/types-of-keras-models#Brain-Tumor-MRI-Segmentation](https://www.kaggle.com/code/abtahiislam/types-of-keras-models#Brain-Tumor-MRI-Segmentation)

---

## Environment & dependencies (quick)

* Python 3.8+ (3.9 or 3.10 recommended)
* TensorFlow 2.x (e.g., `tensorflow>=2.8`) — uses `tf.keras` applications and layers
* Common packages:

  * `numpy`, `pandas`
  * `matplotlib`, `seaborn`
  * `scikit-learn`
  * `opencv-python` (cv2) or `Pillow` for image I/O
  * `tqdm`
    
---

## What I compared / why

For each model I:

* Used a shared helper function to instantiate the pretrained backbone (imagenet weights optional), attach task-specific heads, and compile the model.
* Kept training settings consistent across models where possible (same optimizer, learning-rate schedule, augmentation pipeline) to make comparisons fair.
* Collected: training/validation loss, accuracy (or other segmentation metrics used), training time, and sample qualitative outputs (overlayed predicted segmentation / class maps).
* Ran a “single prediction” demonstration for the best-performing model to show inference code and visual output.

---

## How to interpret results (recommended)

* **Quantitative metrics:** Report accuracy, precision, recall, F1, IoU (for segmentation tasks), and training time per epoch. Put them in a single table for easy comparison.
* **Qualitative:** For segmentation, always check the raw mask overlay on MRI slices — some models have slightly higher numeric scores but produce poorer-looking masks (or more false positives).
* **Cost vs. gain:** Big models (ConvNeXt-XLarge, EfficientNetB7, NASNetLarge) may slightly outperform smaller models but require substantially more GPU time and memory.

---

## Results (template to paste your numbers)
| Model              | Val Accuracy | Val Loss | IoU / F1 | Time / Epoch | Notes |
|--------------------|--------------|----------|----------|--------------|-------|
| Xception           | —            | —        | —        | —            | Lightweight, good feature extraction, stable training |
| VGG19              | —            | —        | —        | —            | High parameters, slower training, prone to overfitting |
| ResNet50V2         | —            | —        | —        | —            | Balanced depth, strong baseline performance |
| ResNet101V2        | —            | —        | —        | —            | Deeper network, improved feature learning |
| ResNet152V2        | —            | —        | —        | —            | Very deep, high compute cost |
| InceptionV3        | —            | —        | —        | —            | Multi-scale feature extraction |
| InceptionResNetV2  | —            | —        | —        | —            | Strong accuracy, heavy architecture |
| MobileNetV2        | —            | —        | —        | —            | Fast and efficient, lower compute |
| DenseNet201        | —            | —        | —        | —            | Strong gradient flow, good feature reuse |
| NASNetLarge        | —            | —        | —        | —            | Auto-designed architecture, very heavy |
| EfficientNetB7     | —            | —        | —        | —            | Excellent accuracy, high memory usage |
| EfficientNetV2-B3  | —            | —        | —        | —            | Faster convergence, balanced performance |
| EfficientNetV2-L   | —            | —        | —        | —            | High accuracy, expensive training |
| ConvNeXt-XLarge    | —            | —        | —        | —            | Modern CNN, transformer-inspired design |
