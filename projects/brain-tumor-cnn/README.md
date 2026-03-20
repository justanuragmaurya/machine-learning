# Brain Tumor MRI Classifier

A web app that classifies brain MRI scans into four categories — **glioma**, **meningioma**, **no tumor**, and **pituitary** — using a custom CNN built with PyTorch.

## Prerequisites

- Python 3.10+
- Node 18+
- CUDA (optional, speeds up training)

## Setup

### 1. Download the dataset

Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and unzip it into `./data/` so the structure looks like:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 2. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This takes ~15–25 minutes on a GPU, or ~60 minutes on CPU. Expected test accuracy: **~82–88%**.

The trained model is saved to `model/best_model.pt` and class names to `model/class_names.json`.

### 4. Start the Flask backend

```bash
cd backend
flask run
```

The API runs on `http://localhost:5000`.

### 5. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown in the terminal (usually `http://localhost:5173`).

## Usage

1. Open the frontend in your browser
2. Upload or drag-and-drop a brain MRI image
3. Click **Classify**
4. View the predicted tumor type with confidence scores and top-3 predictions
