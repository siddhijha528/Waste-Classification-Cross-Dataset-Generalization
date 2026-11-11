**Cross-Dataset Generalization of Deep Learning Models for Waste Image Classification**
**Evidence from TrashNet and RealWaste**
**Authors:** Siddhi Jha, Pooja Gupta  
**Department:** Data Science and Engineering  
**Institution:** Manipal University Jaipur, Jaipur, India

## 1. Abstract
This project investigates the cross-dataset generalization of deep learning-based waste image classifiers trained on TrashNet and evaluated on RealWaste. Five model configurations are examined: Custom CNN, baseline MobileNetV2, MobileNetV2 with standard data augmentation, MobileNetV2 with partial fine-tuning, and MobileNetV2 with strong data augmentation.
The models are trained using the TrashNet dataset and independently evaluated on a stratified held-out subset of RealWaste containing 928 images from five common waste categories.
The experimental results demonstrate that Strong Augmentation MobileNetV2 achieves the highest overall classification performance, while Fine-Tuned MobileNetV2 obtains the highest weighted precision. The study highlights the challenges of transferring models trained on controlled images to real-world waste environments.

## 2. Problem Statement
Deep learning-based waste classification models may achieve good performance on controlled benchmark datasets but experience reduced accuracy when evaluated on images captured under real-world conditions.
Variations in lighting, background, object orientation, and image characteristics create a dataset shift that affects model generalization.
This study investigates the cross-dataset generalization capability of lightweight deep learning models by training on TrashNet and evaluating independently on RealWaste without target-domain training or fine-tuning.

## 3. Objectives
- Develop and evaluate deep learning-based waste classification models.
- Compare Custom CNN and MobileNetV2 architectures.
- Investigate the impact of standard and strong data augmentation.
- Examine the effect of partial fine-tuning.
- Evaluate cross-dataset generalization using an independent target dataset.
- Analyze classification performance using accuracy, precision, recall, F1-score, and confusion matrices.

## 4. Datasets
### 4.1 TrashNet – Source Dataset
TrashNet is used for model training and validation.
- Total images: 2,527
- Number of classes: 6
- Training split: 70%
- Validation split: 15%
- Testing split: 15%
- Random seed: 42

**Classes:**
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash
Dataset: https://github.com/garythung/trashnet
### 4.2 RealWaste – External Target Dataset
RealWaste is used for independent cross-dataset evaluation.
- Selected images: 3,092
- Selected categories: 5
- Development split: 70%
- Held-out test split: 30%
- Held-out test images: 928
- Random seed: 42

**Selected classes:**
- Cardboard
- Glass
- Metal
- Paper
- Plastic
Dataset: https://doi.org/10.24432/C5SS4G
The original six-class model output space is retained during evaluation. Predictions assigned to the unsupported Trash class are counted as incorrect.

## 5. Methodology
The experimental workflow consists of the following stages:
1. Dataset acquisition and organization.
2. Stratified source-domain dataset splitting.
3. Image resizing and model-specific preprocessing.
4. Model construction and training using TrashNet.
5. Model checkpoint saving.
6. Independent evaluation using RealWaste.
7. Class-wise performance analysis.
8. Confusion matrix generation and comparison.

### 5.1 Preprocessing
- Image dimensions: 224 × 224 × 3
- Color format: RGB
- Batch size: 32
- Custom CNN normalization: Pixel values scaled using 1/255.
- MobileNetV2 preprocessing: Pixel values scaled to the range [-1, 1].

### 5.2 Model Configurations
Five model configurations are evaluated:

**1. Custom CNN**
- Three convolutional blocks.
- 32, 64, and 128 filters.
- ReLU activation.
- Max pooling.
- 128-unit dense layer.
- 50% dropout.
- Six-class softmax output.

**2. Baseline MobileNetV2**
- ImageNet-pretrained weights.
- Frozen feature extraction backbone.
- Global average pooling.
- Dropout rate of 0.30.
- Six-class classification head.

**3. Standard Augmentation MobileNetV2**
- Horizontal flipping.
- Rotation factor: 0.1.
- Zoom factor: 0.1.

**4. Fine-Tuned MobileNetV2**
- Initialized from the standard-augmentation model.
- Last 30 backbone layers unfrozen.
- Batch Normalization layers frozen.
- Adam optimizer.
- Learning rate: 1 × 10⁻⁵.
- Five additional epochs.

**5. Strong Augmentation MobileNetV2**
- Horizontal flipping.
- Rotation factor: 0.15.
- Zoom factor: 0.15.
- Contrast factor: 0.2.
- Dropout rate: 0.40.
- 15 training epochs.

## 6. Experimental Results
The following table presents the performance of all five model configurations on the RealWaste held-out test set.
| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Custom CNN | 35.56% | 0.3670 | 0.3556 | 0.3412 |
| Baseline MobileNetV2 | 49.89% | 0.6150 | 0.4989 | 0.5361 |
| Standard Augmentation MobileNetV2 | 46.34% | 0.6549 | 0.4634 | 0.5294 |
| Fine-Tuned MobileNetV2 | 45.47% | 0.6854 | 0.4547 | 0.5207 |
| Strong Augmentation MobileNetV2 | **50.75%** | 0.6235 | 0.5075 | **0.5494** |

### Key Findings
- Strong Augmentation MobileNetV2 achieved the highest accuracy of 50.75%.
- Strong Augmentation MobileNetV2 achieved the highest weighted F1-score of 0.5494.
- Fine-Tuned MobileNetV2 achieved the highest weighted precision of 0.6854.
- Custom CNN recorded the lowest overall accuracy of 35.56%.
- Metal achieved the highest class-wise F1-score (0.70) for Strong Augmentation MobileNetV2.
- Glass and cardboard showed comparatively lower recall.

## 7. Evaluation Metrics
The following evaluation metrics are used:
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Class-wise Precision, Recall, and F1-score
- Confusion Matrix

## 8. Technologies and Tools
- Python
- TensorFlow / Keras
- Google Colab
- MobileNetV2
- Convolutional Neural Networks (CNN)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 9. Experimental Environment
| Configuration | Details |
|---|---|
| Execution Environment | Google Colab |
| Framework | TensorFlow / Keras |
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Source Dataset | TrashNet |
| Target Dataset | RealWaste |

Inference latency, energy consumption, throughput, and memory usage were not measured in this study.

## 10. Limitations
- Each model configuration was trained only once.
- Evaluation uses a single source-data split and one external test partition.

## 11. Future Work
Future research may investigate:
- Multiple training runs with different random seeds.
- Broader domain-diverse training datasets.
- Class-balanced learning strategies.
- Confidence calibration.
- Domain adaptation techniques.
- Model quantization and edge deployment.
- Inference latency, memory usage, and energy consumption.

## 12. Research Contribution
The primary contribution of this study is an empirical evaluation of cross-dataset generalization from TrashNet to RealWaste.
Unlike conventional within-dataset evaluation, this work investigates how different training configurations perform on an independent target dataset without target-domain training or fine-tuning.
The study also retains the complete six-class output space and counts predictions assigned to the unsupported Trash class as incorrect, providing a transparent evaluation protocol.

## 13. Authors
**Siddhi Jha**  
Conceptualization, methodology, implementation, and manuscript preparation.
**Pooja Gupta**  
Supervision, manuscript review, and editing.

Department of Data Science and Engineering  
Manipal University Jaipur, Jaipur, India.

## 14. References

1. TrashNet Dataset: https://github.com/garythung/trashnet
2. RealWaste Dataset: https://doi.org/10.24432/C5SS4G
