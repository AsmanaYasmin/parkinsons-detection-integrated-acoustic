# Integrated Acoustic Feature Analysis for Parkinson’s Disease Detection Using Spectrogram and Formant Images with Deep Feature Extraction and Machine Learning Models

Integrated Acoustic Feature Analysis for Parkinson’s Disease Detection Using Spectrogram and Formant Images with Deep Feature Extraction and Machine Learning Models
GitHub Repository Documentation & Execution Guide (README.md)
1. Overview
Parkinson’s disease (PD) is a chronic and progressive neurodegenerative disorder primarily affecting the motor system, with vocal and speech impairments (hypokinetic dysarthria) frequently emerging as early clinical indicators. Conventional diagnostic procedures depend heavily on specialist-driven clinical assessments and costly neuroimaging techniques (such as DaTscan and MRI), which are often inaccessible in resource-limited and rural healthcare environments.
This repository presents a non-invasive, objective, and low-cost screening framework for Parkinson’s disease detection using acoustic speech recordings from the clinically validated PC-GITA Spanish speech database. Speech signals are transformed into two complementary visual representations: time-frequency Spectrograms and vocal-tract resonance Formant images (F1–F4). Pretrained deep vision backbones (including Vision Transformer, InceptionV3, and DenseNet121) are deployed as feature extractors, followed by classical machine learning classifiers to deliver accurate and robust diagnostic screening.
2. Pipeline Architecture & Workflow
• Acoustic Segmentation:  Continuous raw speech signals (.wav) are split into standardized, non-overlapping 10-second segments to enforce input length consistency.
• Visual Transformation:  Speech segments are converted into 2D STFT magnitude spectrograms and formant trajectory images (F1, F2, F3, F4) via acoustic/phonetic tracking.
• Image Enhancement & Preprocessing:  Grayscale conversion, median/Gaussian filtering, edge detection/blending, log-amplitude dynamic range compression, and Min-Max scaling. Class-wise data augmentation (rotation, flipping, color jitter, resized crop) balances the dataset to 1,000 spectrogram images and 500 formant images per class.
• Deep Feature Extraction:  Ten deep backbones (ViT, InceptionV3, DenseNet121, ResNet50, VGG16, MobileNetV2, EfficientNetB0, AlexNet, Xception, CustomCNN) extract compact, high-level feature embeddings from preprocessed images.
• Classification & Decision:  Deep feature vectors are classified into Parkinson's Disease (PD) and Healthy Control (HC) using nine machine learning classifiers (Logistic Regression, SVM, Random Forest, AdaBoost, Gradient Boosting, Extra Trees, KNN, Decision Tree, Naïve Bayes).
[Raw Audio (.wav) - PC-GITA]
                 │
                 ▼
[10-Second Non-Overlapping Segmentation]
       ├─────────────────────────────────┐
       ▼                                 ▼
[Spectrogram Images]             [Formant Images (F1–F4)]
       │                                 │
       ▼                                 ▼
[Enhancement & Augmentation]     [Enhancement & Augmentation]
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
3. Key Findings & Experimental Results
A comprehensive comparison across representations and classifiers reveals superior performance when pairing deep visual embeddings with lightweight linear classification:
Feature Modality	Best Feature Extractor	Best Classifier	Accuracy	Precision	Recall	F1-Score
Spectrogram	Vision Transformer (ViT)	Logistic Regression	99.00%	97.04%	98.50%	97.77%
Formants (F1+F2+F3)	InceptionV3 / DenseNet121	Logistic Regression	96.19%	96.00%	96.00%	96.19%

Benchmark comparison against prior spectrogram-based studies on the PC-GITA corpus:
Study / Reference	Spectrogram Representation & Model	Classifier	Accuracy (%)
Little et al. (2007)	Spectrogram + CNN	Softmax	91.20%
Wodzinski et al. (2017)	Spectrogram + CNN	Softmax	94.00%
Orozco-Arroyave et al. (2014)	Log-Spectrogram + CNN	Softmax	95.10%
Vásquez-Correa et al. (2019)	Mel-Spectrogram + CNN	SVM	95.80%
Malekroodi et al. (2025)	Multi-channel Spectrogram + Deep CNN	Softmax	96.50%
Gunduz (2019)	Spectrogram + CNN + LSTM	Softmax	96.90%
Lee et al. (2024)	Audio Spectrogram Transformer (AST)	End-to-End	97.80%
Proposed Framework	Enhanced Spectrogram + Vision Transformer (ViT)	Logistic Regression	99.00%

4. Dataset Citation & Setup
The experiments utilize the clinically validated PC-GITA Spanish speech corpus collected at Clínica Noel (Medellín, Colombia):
• Subject Distribution:  100 native Colombian Spanish speakers (50 Parkinson's Disease patients, 50 Healthy Controls), balanced across gender (25 men, 25 women in each group) and age.
• Acoustic Conditions:  Noise-controlled soundproof booth, omnidirectional microphone (Shure SM63L), sampled at 44.1 kHz with 16-bit resolution.
• Speech Modalities:  Vowels, diadochokinetic (DDK) syllables, read sentences, and monologues.
*Note: Due to privacy regulations and file hosting constraints, raw audio recordings are not directly distributed in this repository. Researchers can request access through the official PC-GITA repository.*
5. How to Run (Step-by-Step Execution Guide)
This repository contains all experimental notebooks and full thesis documentation. Follow the steps below to execute the pipeline:
Step 1: Environment Setup & Dependencies
Clone the repository and install the required Python packages in a Python 3.10+ environment:
# Clone this repository
git clone https://github.com/AsmanaYasmin/parkinsons-detection-integrated-acoustic.git
cd parkinsons-detection-integrated-acoustic

# Install dependencies
pip install torch torchvision torchaudio scikit-learn librosa praat-parselmouth numpy pandas matplotlib seaborn

Step 2: Repository Structure Overview
The repository structure corresponds directly to your uploaded files:
• `colab_file/`: Contains the Google Colab / Jupyter Notebooks for data preprocessing, image transformation, and model evaluation.
• `Thesis report (Parkinson's Disease).pdf`: Complete undergraduate research thesis report.
• `Thesis report (Parkinson's Disease).docx`: Editable manuscript version of the thesis.
• `presentation.pdf`: Defense slide deck summarizing the research methodology and findings.
• `README.md`: Project summary, architecture, experimental results, and run commands.
Step 3: Executing via Google Colab / Jupyter Notebook
Open the notebooks located inside the `colab_file/` folder in Google Colab or JupyterLab:
1. Mount Google Drive and specify the path to your preprocessed spectrogram and formant folders.
2. Execute the notebook sections in sequence:
   • Section 1 (Imports): PyTorch, torchvision, torchaudio, and scikit-learn models.
   • Section 2 (Audio Preprocessing): 10-second non-overlapping segmentation and visual generation (STFT & Praat formants).
   • Section 3 (Image Enhancement & Augmentation): Median/Gaussian filtering, edge detection, and PyTorch torchvision transforms.
   • Section 4 (Deep Feature Extraction): Extracts high-dimensional embeddings using ViT, InceptionV3, and DenseNet121.
   • Section 5 (Classification & Metrics): Trains Logistic Regression, SVM, and tree ensembles, generating accuracy, confusion matrices, and ROC-AUC curves.
Step 4: Verification & Output Reproducibility
The models will validate predictions on the held-out test split, confirming 99.00% accuracy on ViT + Logistic Regression and 96.19% accuracy on Formant (F1+F2+F3) + InceptionV3 + Logistic Regression.

