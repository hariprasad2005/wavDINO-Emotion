# Orchestrates the full wavDINO-Emotion pipeline on Windows PowerShell.
# Assumes Python dependencies (torch, torchaudio, torchvision, pillow, numpy) are installed.

param(
    [string]$Python = "G:/paper/.venvcuda/Scripts/python.exe",
    [string]$RepoRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

# Dataset roots (override if needed)
$crema = "wavDINO-Emotion/datasets/CREMA-D-REDUCED"
$ravdess = "wavDINO-Emotion/datasets/RAVDESS_CLASSIFIED"
$afew = "wavDINO-Emotion/datasets/AFEW_CLASSIFIED"

Write-Host "[1/6] Generating splits (AFEW capped to 1000/label)" -ForegroundColor Cyan
& $Python src/preprocess.py --crema_root $crema --ravdess_root $ravdess --afew_root $afew --output_dir "./splits" --afew_fraction 1.0 --afew_cap_per_label 1000

Write-Host "[2/6] Extracting audio embeddings" -ForegroundColor Cyan
& $Python src/extract_audio.py --csv "./splits/crema_train.csv" --output "./embeddings/audio/crema_train.npy"
& $Python src/extract_audio.py --csv "./splits/crema_val.csv" --output "./embeddings/audio/crema_val.npy"
& $Python src/extract_audio.py --csv "./splits/crema_test.csv" --output "./embeddings/audio/crema_test.npy"
& $Python src/extract_audio.py --csv "./splits/ravdess_train.csv" --output "./embeddings/audio/ravdess_train.npy"
& $Python src/extract_audio.py --csv "./splits/ravdess_val.csv" --output "./embeddings/audio/ravdess_val.npy"
& $Python src/extract_audio.py --csv "./splits/ravdess_test.csv" --output "./embeddings/audio/ravdess_test.npy"

Write-Host "[3/6] Extracting visual embeddings" -ForegroundColor Cyan
& $Python src/extract_visual.py --csv "./splits/afew_train.csv" --output "./embeddings/visual/afew_train.npy"
& $Python src/extract_visual.py --csv "./splits/afew_val.csv" --output "./embeddings/visual/afew_val.npy"
& $Python src/extract_visual.py --csv "./splits/afew_test.csv" --output "./embeddings/visual/afew_test.npy"

Write-Host "[4/6] Training per-dataset fusion models (TABLE II)" -ForegroundColor Cyan
& $Python src/train_fusion.py --train_audio "./embeddings/audio/crema_train.npy" --val_audio "./embeddings/audio/crema_val.npy" --log_path "./logs/training_crema.txt" --model_path "./models/fusion_crema.pt" --epochs 150 --lr 1e-4 --batch_size 8 --cosine --balance
& $Python src/train_fusion.py --train_audio "./embeddings/audio/ravdess_train.npy" --val_audio "./embeddings/audio/ravdess_val.npy" --log_path "./logs/training_ravdess.txt" --model_path "./models/fusion_ravdess.pt" --epochs 150 --lr 1e-4 --batch_size 8 --cosine --balance
& $Python src/train_fusion.py --train_visual "./embeddings/visual/afew_train.npy" --val_visual "./embeddings/visual/afew_val.npy" --log_path "./logs/training_afew.txt" --model_path "./models/fusion_afew.pt" --epochs 150 --lr 1e-4 --batch_size 8 --cosine --balance

Write-Host "[5/6] Evaluating single-dataset performance (TABLE II)" -ForegroundColor Cyan
& $Python src/evaluate.py --audio "./embeddings/audio/crema_test.npy" --model "./models/fusion_crema.pt" --dataset_name "CREMA-D" --table_path "./results/tables/table_II_performance.csv"
& $Python src/evaluate.py --audio "./embeddings/audio/ravdess_test.npy" --model "./models/fusion_ravdess.pt" --dataset_name "RAVDESS" --table_path "./results/tables/table_II_performance.csv"
& $Python src/evaluate.py --visual "./embeddings/visual/afew_test.npy" --model "./models/fusion_afew.pt" --dataset_name "AFEW" --table_path "./results/tables/table_II_performance.csv"

Write-Host "[6/6] Cross-dataset evaluation (TABLE I)" -ForegroundColor Cyan
$manifest = @{ datasets = @(
    @{ name = "CREMA-D"; audio_train = "./embeddings/audio/crema_train.npy"; audio_val = "./embeddings/audio/crema_val.npy"; audio_test = "./embeddings/audio/crema_test.npy" },
    @{ name = "RAVDESS"; audio_train = "./embeddings/audio/ravdess_train.npy"; audio_val = "./embeddings/audio/ravdess_val.npy"; audio_test = "./embeddings/audio/ravdess_test.npy" },
    @{ name = "AFEW"; visual_train = "./embeddings/visual/afew_train.npy"; visual_val = "./embeddings/visual/afew_val.npy"; visual_test = "./embeddings/visual/afew_test.npy" }
) } | ConvertTo-Json -Depth 4
$manifestPath = Join-Path $RepoRoot "manifest.json"
$manifest | Out-File -FilePath $manifestPath -Encoding utf8
& $Python src/cross_dataset_eval.py --manifest $manifestPath --output "./results/tables/table_I_cross_dataset.csv" --epochs 60 --batch_size 16 --lr 3e-4 --balance

Write-Host "Pipeline finished. Outputs are under results/, logs/, models/, and embeddings/." -ForegroundColor Green
