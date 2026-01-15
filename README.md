# Next-Gen Surveillance: AI Agent for Violence Detection & Emergency Alerts

An AI-driven CCTV surveillance prototype that detects **violent vs non-violent** activity from short video clips using a **MobileNetV2** deep learning model and triggers automated alerts via **Telegram** (message + evidence video) and **Twilio** (phone call) when violence is detected.

> **Note:** Dataset is from Kaggle and is **not included** in this repository.

---

## What this repo contains

### Model
- `models/violence_mobilenetV2_FINAL.keras` — trained MobileNetV2-based classifier (~22.7MB)

### Notebooks
- `notebooks/violence detection.ipynb`  
  Data integration, preprocessing, model building, training, evaluation, and saving the model.
- `notebooks/violence detection keras.ipynb`  
  Loads the saved `.keras` model, performs prediction on an uploaded/test video, and triggers alerting (Telegram + Twilio).

### Optional folders (recommended)
- `assets/` — screenshots (architecture diagram, confusion matrix, metrics table, sample frames)
- `data/` — local-only dataset folder (ignored by `.gitignore`)

---

## System overview (high-level pipeline)

1. **Input video** (CCTV clip / uploaded video)
2. **Preprocessing**
   - Extract frames (default: **5 frames per video**)
   - Convert BGR → RGB
   - Resize to **112 × 112**
   - Normalize pixel values to **[0,1]**
3. **Model inference**
   - MobileNetV2 predicts violence probability per frame
   - Average probability across frames → final score
4. **AI Agent decision**
   - If score ≥ threshold (default: **0.5**) → **VIOLENCE**
   - Else → **NON-VIOLENCE**
5. **Evidence + Alerts**
   - Evidence clip (short segment / original video, depending on notebook logic)
   - Telegram notification + video
   - Twilio phone call alert

---

## Results (from project evaluation)

- Validation Accuracy: **~92%**
- Balanced performance across both classes (Precision/Recall/F1 ~0.92 overall)
- Confusion matrix shows low misclassification compared to correct predictions

---

## Requirements

- Python **3.x**
- TensorFlow/Keras
- OpenCV
- NumPy, scikit-learn
- Telegram Bot API
- Twilio API
- (Optional) Gradio UI

Install dependencies:
```bash
pip install -r requirements.txt
