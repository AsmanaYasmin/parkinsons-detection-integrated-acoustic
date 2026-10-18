# Integrated Acoustic Feature Analysis for Parkinson’s Disease Detection Using Spectrogram and Formant Images with Deep Feature Extraction and Machine Learning Models

## 1. Overview
Parkinson’s disease (PD) is a chronic and progressive neurodegenerative disorder primarily affecting the motor system, with vocal and speech impairments (hypokinetic dysarthria) frequently emerging as early indicators. Conventional clinical diagnostics rely heavily on specialist-driven assessments or expensive neuroimaging techniques (e.g., MRI, DaTscan), which remain largely inaccessible in resource-limited and rural healthcare settings.

This project presents a non-invasive, low-cost screening framework utilizing speech recordings from the clinically validated **PC-GITA** dataset. Audio signals are converted into complementary visual representations: **time-frequency Spectrograms** and **vocal-tract resonance Formants (F1–F4)**. By leveraging specialized image preprocessing, class-balancing data augmentation, deep feature extraction via pretrained backbones, and classical machine learning classifiers, this framework delivers highly reliable diagnostic screening[cite: 54].

---

## 2. Pipeline Architecture
The proposed system follows an end-to-end modular pipeline[cite: 54]:

1. **Acoustic Segmentation:** Raw `.wav` speech recordings are segmented into standardized 10-second non-overlapping windows[cite: 54].
2. **Visual Transformations:**
   - **Spectrogram Images:** Extracted using Short-Time Fourier Transform (STFT) magnitude maps with dynamic range compression[cite: 54].
   - **Formant Images (F1–F4):** Articulatory resonance trajectories extracted across fundamental formant tracks[cite: 54].
3. **Image Enhancement & Preprocessing:**
   - Single-channel grayscale conversion, median/Gaussian filtering for noise suppression, edge enhancement (Canny/FIND_EDGES), and Min-Max scaling[cite: 54].
   - Data augmentation (rotation, horizontal/vertical flips, color jitter, and resized crops) to enforce balanced training distributions[cite: 54].
4. **Deep Feature Extraction:**
   - High-level representation extraction using 10 backbones: Vision Transformer (ViT), InceptionV3, DenseNet121, ResNet50, VGG16, MobileNetV2, EfficientNetB0, AlexNet, Xception, and CustomCNN[cite: 54].
5. **Machine Learning Classification:**
   - Prediction across deep embeddings using 9 classical classifiers: Logistic Regression, Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN), Decision Tree, Gaussian Naive Bayes, AdaBoost, Gradient Boosting, and Extra Trees[cite: 54].

```text
[Raw Audio (.wav)] 
       │
       ▼
[10-Second Segmentation]
       ├─────────────────────────────────┐
       ▼                                 ▼
[Spectrogram Images]             [Formant Images (F1–F4)]
       │                                 │
       ▼                                 ▼
[Image Enhancement & DL Norm]    [Image Enhancement & DL Norm]
       │                                 │
       ▼                                 ▼
[ViT Feature Extraction]         [InceptionV3 Feature Extraction (F1+F2+F3)]
       │                                 │
       ▼                                 ▼
[Logistic Regression]            [Logistic Regression]
       │                                 │
       └────────────────┬────────────────┘
                        ▼
                 [Prediction: PD vs HC]

## 3. Key Findings

### Best-Performing Configurations
| Feature Representation | Feature Extractor | Best Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Spectrogram** | **Vision Transformer (ViT)** | **Logistic Regression** | **99.00%** | **97.04%** | **98.50%** | **97.77%** |
| **Formants (F1+F2+F3)** | **InceptionV3** | **Logistic Regression** | **96.19%** | **96.00%** | **96.00%** | **96.19%** |

*Both models achieved high discriminative separation, reflected in the Receiver Operating Characteristic (ROC) analysis with near-ideal Area Under the Curve (AUC) metrics.*

### Comparative Benchmark on the PC-GITA Dataset
| Study | Architecture / Method | Classifier | Accuracy (%) |
| :--- | :--- | :---: | :---: |
| Little et al. | CNN | Softmax | 91.20% |
| Wodzinski et al. | CNN | Softmax | 94.00% |
| Orozco-Arroyave et al. | Log Spectrogram + CNN | Softmax | 95.10% |
| Vásquez-Correa et al. | Mel Spectrogram + CNN | SVM | 95.80% |
| Malekroodi et al. | Multi-channel Spectrogram + Deep CNN | Softmax | 96.50% |
| Gunduz | Spectrogram + CNN + LSTM | Softmax | 96.90% |
| Lee et al. | Audio Spectrogram Transformer (AST) | End-to-End | 97.80% |
| **Proposed Framework** | **Spectrogram + Vision Transformer (ViT)** | **Logistic Regression** | **99.00%** |

---

## 4. Dataset Citation
This research evaluates recordings from the **PC-GITA** Spanish speech database:
- **Subjects:** 100 participants (50 Parkinson’s disease patients, 50 Healthy Controls), gender- and age-balanced (25 males, 25 females per group).
- **Acoustic Setup:** Recorded in a noise-controlled soundproof booth at 44.1 kHz, 16-bit resolution using a dynamic omnidirectional microphone.

```bibtex
@article{orozco2014automatic,
  title={Automatic detection of Parkinson's disease in running speech spoken in three different languages},
  author={Orozco-Arroyave, Juan Rafael and H{\"o}nig, Florian and Arias-Londo{\~n}o, Juli{\'a}n David and others},
  journal={The Journal of the Acoustical Society of America},
  volume={136},
  number={4},
  pages={2175--2175},
  year={2014}
}

Due to license restrictions and file size, the PC-GITA raw audio dataset is not hosted in this repository. Researchers can request access from the original dataset providers
