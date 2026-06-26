# SafeCart: AI-Powered Marketplace Listing Validator

An AI-powered system that detects suspicious or potentially fraudulent e-commerce product listings using Natural Language Processing (NLP), numerical feature analysis, and image-text similarity.

The project combines sentence embeddings with structured product features and a deep learning classifier to assign a suspicion score for marketplace listings.

---

## Features

- Detects suspicious marketplace listings
- Uses NLP for title and description analysis
- Incorporates image-text similarity
- Uses pricing anomaly detection
- Deep Learning based classification
- Returns probability (Suspicion Score)
- Easy prediction pipeline for new products

---

## Dataset

The project uses an Amazon India product listing dataset containing:

- Product Title
- Product Description
- Category
- Price
- MSRP
- Product Image
- Image URL

Additional engineered features:

- Image-Text Similarity (CLIP)
- Price Anomaly Z-Score
- Binary Suspicious Label

---

## Project Structure

```
├── Build_hybrid_model.ipynb          # Model training
├── Prediction.ipynb                  # Prediction pipeline
├── Image_preprocessing.ipynb         # Image feature extraction
├── create_base_features_with_image.ipynb
├── base_features_with_image.csv
├── suspicion_model.pt
├── scaler.pkl
├── pca_transform.pkl
└── README.md
```

---

## Model Pipeline

### 1. Data Preprocessing

- Missing value handling
- Feature engineering
- Text cleaning
- Price normalization

---

### 2. Text Embedding

Sentence embeddings generated using

- **all-MiniLM-L6-v2 (Sentence Transformers)**

Input Text:

```
Title + Description + Category
```

---

### 3. Dimensionality Reduction

- PCA
- Reduced embeddings to 64 dimensions

---

### 4. Numerical Features

The model also learns from:

- Price
- MSRP
- Price Anomaly Z-Score
- CLIP Image-Text Similarity

These features are standardized using StandardScaler.

---

### 5. Hybrid Feature Vector

Final input to the neural network:

```
PCA(Text Embedding)
        +
Numerical Features
        ↓
Hybrid Feature Vector
```

---

### 6. Deep Learning Model

Feed Forward Neural Network (PyTorch)

Architecture:

```
Input
   ↓
Linear Layer
   ↓
ReLU
   ↓
Dropout
   ↓
Linear Layer
   ↓
Sigmoid
```

Output:

```
Suspicion Score (0 - 1)
```

Threshold:

```
0.65
```

---

## Technologies Used

- Python
- PyTorch
- Sentence Transformers
- Scikit-learn
- Pandas
- NumPy
- PCA
- StandardScaler

---

## Installation

```bash
git clone https://github.com/RajdeepT/safecart-ai.git

cd safecart-ai

pip install -r requirements.txt
```

---

## Required Libraries

```bash
pip install torch
pip install sentence-transformers
pip install scikit-learn
pip install pandas
pip install numpy
```

---

## Running the Project

### Train the Model

```
Build_hybrid_model.ipynb
```

This notebook:

- Loads dataset
- Creates embeddings
- Performs PCA
- Trains the neural network
- Saves the model

Outputs:

```
suspicion_model.pt
pca_transform.pkl
scaler.pkl
```

---

### Prediction

Run:

```
Prediction.ipynb
```

Input example:

```python
record = {
    "title": "...",
    "description": "...",
    "category": "...",
    "price": ...,
    "msrp": ...,
    "price_anomaly_z": ...,
    "clip_image_text_sim": ...
}
```

Output:

```
Suspicion Score: 0.87

Prediction:
Suspicious Listing
```

---

## Applications

- E-commerce fraud detection
- Fake product identification
- Marketplace moderation
- Trust & Safety systems
- Seller quality monitoring

---

## Future Improvements

- Fine-tune BERT models
- Vision Transformer (ViT) integration
- Real-time API deployment
- Explainable AI (SHAP/LIME)
- Streamlit or FastAPI dashboard
- Multi-class fraud categorization

---

## Author

**Rajdeep Thakur**

B.Tech in Information Technology  
KIIT University

AI • Machine Learning • Data Analytics
