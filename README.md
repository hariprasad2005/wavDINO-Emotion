# wavDINO-Emotion

Reproducible pipeline for audio-visual emotion recognition with Wav2Vec2 + DINOv2 encoders and a fusion Transformer. All outputs (splits, embeddings, models, logs, tables, figures) are saved under this directory for auditability.

## Expected dataset layout
- G:/CREMA-D-CLASSIFIED/<label>/*.wav
- G:/ravdess1/RAVDESS_CLASSIFIED/<label>/*.wav
- G:/afew1/AFEW_CLASSIFIED/<label>/*.(jpg|png)

Labels (exactly 5): angry, happy, sad, neutral, surprise.

## Pipeline steps
1. Generate 70/15/15 splits with random_state=42 → `splits/`.
2. Extract 1024-d audio embeddings (Wav2Vec2) and visual embeddings (DINOv2) → `embeddings/{audio,visual}/`.
3. Train fusion Transformer (AdamW, lr=3e-4, epochs=50) → `models/` + `logs/training_log.txt`.
4. Evaluate single-dataset (TABLE II) and cross-dataset (TABLE I) → `results/tables/` + `logs/cross_dataset_log.txt`.
5. Generate plots (loss, accuracy, t-SNE, attention, confusion) → `results/figures/`.

## Running
Use `run_pipeline.ps1` (added in this repo) to execute end-to-end on Windows PowerShell. Configure paths and environment variables inside the script as needed.
