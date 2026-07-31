# ML-algorithm-reproduction

A repository for machine learning study notes and task implementations. Here we not only reproduce classic machine learning algorithms, but also record the ideas and insights gained while learning and turn them into working code. The project is updated continuously as new concepts are explored.

## Overview

The repository is organized into three directories by learning topic, covering machine learning algorithm reproduction, PyTorch framework practice, and Kaggle competition tasks. Most scripts use synthetic data or a public dataset, train a simple model, and visualize the results so the learning process is easy to inspect.

## Repository Structure

```
.
├── Algorithm/  # Classic machine learning algorithm reproductions
│   ├── Linear_Regression.py          # From-scratch linear regression (gradient descent)
│   ├── Logistic_Regression.py        # Logistic regression binary classification experiment
│   ├── Traffic_Flow_Prediction.py    # Neural network regression (traffic volume prediction)
│   ├── Wine_Quality.py               # PyTorch binary classification on the UCI wine quality dataset
│   ├── auto_stack.py                 # Multi-label text classification (TF-IDF + neural network)
│   └── K_fold_cross_validation.py    # K-fold cross-validation
├── pytorch/    # PyTorch framework study notes and exercises
│   ├── Tensors.py                    # Tensor basics
│   └── Quickstart.py                 # Official Quickstart complete training workflow
└── Kaggle/     # Kaggle competition task practice
    └── task-tianic.ipynb             # Titanic survival prediction
```

## What This Project Focuses On

- Reproducing core machine learning ideas in a clear and compact way.
- Understanding the full workflow from data preparation to training and evaluation.
- Recording insights gained during study and turning them into executable experiments.
- Iterating on basic models before moving to larger or more realistic projects.

## Requirements

The examples rely on the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `torchvision`

Some scripts download data from the internet, so an active network connection may be required.

## Installation

If you are using a virtual environment, activate it first, then install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

## Usage

Each file can be run independently:

```bash
python Algorithm/Linear_Regression.py
python Algorithm/Logistic_Regression.py
python Algorithm/Traffic_Flow_Prediction.py
python Algorithm/Wine_Quality.py
python Algorithm/auto_stack.py
python Algorithm/K_fold_cross_validation.py
```

Scripts under `pytorch/` are run the same way. Tasks under `Kaggle/` are Jupyter notebooks and can be opened with:

```bash
jupyter notebook Kaggle/task-tianic.ipynb
```

Most scripts will print training progress to the console and display plots for model behavior, loss curves, or prediction results.

## Notes

- The repository is for study and experimentation, so the scripts are intentionally compact and easy to modify.
- Some examples use synthetic data for clarity rather than realism.
- Some comments in the code are written in Chinese to record study notes. Naming and implementation style may evolve as new machine learning ideas are added.

## Future Direction

Potential next steps for the repository include:

- Adding more classical machine learning algorithms.
- Improving data preprocessing and evaluation pipelines.
- Organizing examples by topic or difficulty.
- Adding reusable utilities for training, plotting, and metrics.

## License

No license has been specified yet. Add one if you plan to share or reuse the code publicly.
