# Disease Classification Text Analytics Dashboard

## Overview

This project presents a comprehensive approach to disease classification using textual features extracted from clinical data. It explores two encoding methods—TF-IDF and one-hot encoding—to represent data extracted from disease risk factors, symptoms, signs, and subtypes. The project incorporates:
- **TF-IDF Feature Extraction:** Parsing stringified lists, converting lists to space‐separated strings, vectorizing using `TfidfVectorizer`, and comparing the resulting TF-IDF matrix against a provided one-hot encoded matrix (with respect to sparsity and unique features).
- **Dimensionality Reduction:** Application of PCA and Truncated SVD on both TF-IDF and one-hot encoded matrices to reduce dimensions to 2–3 components, along with 2D visualizations that color-code diseases by category.
- **Model Evaluation:** Training and evaluating KNN models (with k = 3, 5, 7 and using Euclidean, Manhattan, and Cosine similarity) and Logistic Regression models through 5-fold cross-validation. The project compares performance metrics such as Accuracy, Precision, Recall, and F1-score.
- **Critical Analysis:** A separate critical analysis report discusses why TF-IDF might outperform one-hot encoding (or vice versa) for this dataset, the clinical relevance of the findings, and the limitations of both encoding methods.

In addition, an interactive Streamlit application is provided for KNN-based disease prediction.

## Repository Contents

- **Jupyter Notebook (`Disease_Classification_Dashboard.ipynb`):**  
  Contains the complete implementation divided by tasks:
  - Task 1: Data Preprocessing and TF-IDF Feature Extraction.
  - Task 2: Dimensionality Reduction (PCA and SVD) and Visualization.
  - Task 3: Model Training and Evaluation (KNN and Logistic Regression).
  - Task 4: Critical Analysis (or a reference to the separate analysis document).
  
- **Streamlit Application (`streamlit_app.py`):**  
  An interactive dashboard for disease classification using KNN with TF-IDF features. This app allows users to:
  - Configure KNN parameters (neighbors and distance metrics).
  - Upload or input new textual data for prediction.
  - View predictions from the trained model.

- **Data Files:**
  - `disease_features.csv` – The dataset containing disease textual features.
  - `encoded_output2.csv` – The provided one-hot encoded matrix.

- **Critical Analysis Document (`Critical_Analysis.docx`):**  
  A separate Word file that contains the critical analysis of the encoding methods and model evaluation.

- **Additional Files:**
  - Generated visualizations (e.g., `tfidf_pca_vis.png`, `onehot_svd_vis.png`, etc.)
  - Model performance summary and recommendations (e.g., `model_summary_and_recommendations.md`, `model_results_loo.csv`)

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AbbasHafeez/DiseaseClassification_TextAnalytics.git
   cd DiseaseClassification_TextAnalytics
