<!-- SHOWCASE: true -->

# Impact of Data Augmentation and Regularization On Image Denoising

> A CNN-based image denoising framework that systematically evaluates the impact of data augmentation and regularization strategies across multiple datasets and noise levels.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Spring%202025-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Intro to Neural Networks Deep Learning                                                                                                                                                        |
| Course Number          | EEL6812                                                                                                                                                                                       |
| Semester               | Spring 2025                                                                                                                                                                                   |
| Assignment Title       | Final Project                                                                                                                                                                                 |
| Assignment Description | Design and evaluate a deep learning system on a self-selected research topic, demonstrating understanding of model architectures and training strategies applied to a meaningful application. |

---

## Project Description

This project investigates how data augmentation and regularization techniques affect the performance of a convolutional neural network trained for image denoising. A baseline encoder-decoder CNN was trained on noisy-clean image pairs drawn from MNIST, CIFAR-10, CIFAR-100, and STL-10, across three Gaussian noise levels (15%, 25%, 50%). Each augmentation (brightness, color jitter, contrast, cutout, flipping, Gaussian noise, random crop, rotation, scaling, shearing, and a custom chained method) and each regularization strategy (L1, L2, dropout, and early stopping) was evaluated in isolation to isolate its contribution. Results show that cutout and Gaussian noise augmentation consistently outperform spatial transformations, and that early stopping delivers the strongest regularization benefit for pixel-wise denoising tasks.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

When training completes, results are written to a subdirectory under `Results/` named after the experiment and dataset. Each run produces:

```
Results/
└── base_mnist/
    ├── trainingLog.txt      # Per-epoch train/val loss and accuracy
    ├── results.json         # Final test loss, test accuracy, computation time
    ├── epoch_data.csv       # Full epoch-by-epoch metrics table
    └── visual_comparisons/
        ├── 1_original.png   # Ground-truth clean image
        ├── 1_generated.png  # Model's denoised output
        └── ...
```

**Sample terminal output (end of training):**

```
Epoch 50/50
  Train Loss:         0.0021    Train Accuracy:     0.9612
  Validation Loss:    0.0018    Validation Accuracy: 0.9589
--------------------------------------------------

Final Test Results:
Test Loss: 0.0014
Test Accuracy: 0.9610
Total Computation Time: 578.34 seconds
```

**Interpreting results:**

- **Test accuracy** is computed as the fraction of pixel predictions within 0.1 of the ground-truth value. Scores above 0.90 on MNIST and above 0.75 on STL-10 at 50% noise represent strong denoising.
- **Test loss** (MSE) should decrease steadily across epochs. Loss plateauing early or rising on validation may indicate overfitting; consider reducing regularization strength or enabling early stopping.
- **Visual comparisons** in `visual_comparisons/` provide a qualitative sanity check. If denoised outputs look blurry or washed out, try reducing the noise factor or adjusting the learning rate.
- If results are missing, confirm the `--output_dir` path is writable and that the dataset downloaded correctly to `./data/`.

---

## Key Concepts

`Convolutional Neural Networks` `Image Denoising` `Residual Learning` `Data Augmentation` `L1 Regularization` `L2 Regularization` `Dropout` `Early Stopping` `MSE Loss` `Encoder-Decoder Architecture`

---

## Languages & Tools

- **Language:** Python 3.8+
- **Framework:** PyTorch, torchvision
- **Build System:** pip / requirements.txt

---

## File Structure

```
.
├── Main.py               # Entry point: argument parsing, dataset prep, experiment dispatch
├── Model.py              # CNN encoder-decoder architecture with optional L1/L2/dropout reg
├── Train.py              # Trainer class and DataAugmentationTechniques collection
├── Analyze/              # Post-training analysis and visualization scripts
│   ├── AnalyzeAug.py     # Plots and CSVs for augmentation experiment results
│   ├── AnalyzePreProcess.py  # Pre-processing analysis
│   └── AnalyzeReg.py     # Plots and CSVs for regularization experiment results
├── Results/              # Auto-generated output directories per experiment run
├── data/                 # Auto-downloaded dataset files (MNIST, CIFAR-10, etc.)
├── requirements.txt      # Third-party dependencies
└── README.md             # Project documentation
```

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- pip
- CUDA-capable GPU (optional but recommended; CPU fallback is automatic)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/alexneilgreen/UCF-IntrotoNeuralNetworksDeepLearning-ImageDenoising.git
cd UCF-IntrotoNeuralNetworksDeepLearning-ImageDenoising

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a base experiment
python Main.py --experiment base --dataset MNIST --epochs 50 --learning_rate 0.001
```

### Controls

| Argument          | Values                                           | Description                                |
| ----------------- | ------------------------------------------------ | ------------------------------------------ |
| `--experiment`    | `base`, `all`, `combo`, or any augmentation name | Experiment type to run                     |
| `--dataset`       | `MNIST`, `CIFAR10`, `CIFAR100`, `STL10`          | Dataset to train on                        |
| `--epochs`        | integer (default: `50`)                          | Number of training epochs                  |
| `--learning_rate` | float (default: `0.001`)                         | Adam optimizer learning rate               |
| `--noise`         | float in [0.0, 1.0] (default: `0.5`)             | Gaussian noise intensity applied to inputs |
| `--output_dir`    | string (default: `Results`)                      | Base folder for output files               |
| `--regVal`        | `L1`, `L2`, `DR`, `ES`, `all`, `None`            | Regularization strategy                    |
| `--L1`            | `1e-5`, `1e-4`, `1e-3` (default: `1e-5`)         | L1 penalty strength                        |
| `--L2`            | `1e-4`, `1e-3`, `1e-2` (default: `1e-4`)         | L2 penalty strength                        |
| `--dropRate`      | `0.2`, `0.3`, `0.4` (default: `0.2`)             | Dropout rate (used with `--regVal DR`)     |
| `--ESP`           | `5`, `10`, `15` (default: `10`)                  | Early stopping patience in epochs          |

**Example commands:**

```bash
# Train with Gaussian noise augmentation on CIFAR-10
python Main.py --experiment gaussian_noise --dataset CIFAR10 --epochs 50

# Train with L2 regularization
python Main.py --experiment base --regVal L2 --L2 1e-4 --dataset MNIST

# Run all augmentations sequentially
python Main.py --experiment all --dataset CIFAR10 --epochs 50

# Run the best-performing combo (Cutout + Gaussian Noise + Early Stopping)
python Main.py --experiment combo --dataset STL10

# Analyze results after training
cd Analyze
python AnalyzeAug.py
python AnalyzePreProcess.py
python AnalyzeReg.py
```

---

## Contributors

| Name             | Role                                                               | GitHub                                         |
| ---------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| Alexander Green  | Model architecture, training pipeline, augmentation implementation | [@AlexGreen](https://github.com/AlexGreen)     |
| Joshua Glaspey   | Dataset preparation, regularization experiments, analysis scripts  | [@JoshGlaspey](https://github.com/JoshGlaspey) |
| Arthur Radulescu | Test and Review, Writing Paper                                     | [@username](https://github.com/username)       |

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.
