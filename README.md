# Quality Inspection Using Machine Learning and Computer Vision
## Unsupervised Anomaly Detection for Industrial Quality Control

This project implements an unsupervised anomaly detection system for surface defect detection on pharmaceutical vials. The models are trained only on normal (good) images, meaning no defect labels are required during training.

The system is designed for industrial quality control scenarios where defective samples are rare or unavailable during training.

Three state-of-the-art anomaly detection models are implemented:

- PatchCore
- EfficientAD
- PaDiM

The repository also includes an interactive web demo built with Streamlit, along with evaluation tools and visualization utilities.

---

## Project Features

- Unsupervised training using only normal samples
- Implementation of three modern anomaly detection methods
- Interactive Streamlit application for real-time inference
- Compatibility with MVTec AD-style dataset structure
- Image-level and pixel-level evaluation metrics
- Visualization of anomaly heatmaps, contours, ROC curves, and score distributions

---

## Project Structure

quality_inspection_using_ml_and_cv/

├── .gitignore  
├── LICENSE  
├── README.md  
├── requirements.txt  

├── dataset.py  
├── train.py  
├── train_padim.py  
├── evaluate.py  
├── evaluate_padim.py  
├── demo_app.py  
├── visualize.py  
├── visualize_padim.py  

├── patchcore.py  
├── efficientad.py  
├── padim.py  
├── metrics.py  

├── checkpoints/   (trained model weights, generated after training)  
└── results/       (evaluation outputs and visualizations, generated during experiments)

---

## Quick Start

### 1. Clone the repository

git clone https://github.com/AlidarAsset/quality_inspection_using_ml_and_cv.git  
cd quality_inspection_using_ml_and_cv  

---

### 2. Install dependencies

pip install -r requirements.txt  

---

### 3. Dataset preparation

The dataset should follow an MVTec AD-style structure:

data/vial/

├── train/
│   └── good/              (normal images only for training)

├── test_public/
│   ├── good/              (normal test images)
│   ├── bad/               (anomalous test images)
│   └── ground_truth/
│       └── bad/           (segmentation masks for anomalies)

---

### 4. Training

python train.py --data_path "data/vial"  
python train_padim.py --data_path "data/vial"  

---

### 5. Evaluation

python evaluate.py --data_path "data/vial"  
python evaluate_padim.py --data_path "data/vial"  

---

### 6. Run interactive demo

streamlit run demo_app.py  

The application will be available at:

http://localhost:8501

---

## Author

Alidar Asset  
GitHub: https://github.com/AlidarAsset  

---

## License

This project is licensed under the MIT License.