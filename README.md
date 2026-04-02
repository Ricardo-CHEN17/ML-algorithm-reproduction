# ML-algorithm-reproduction

This repository is a learning notebook for machine learning algorithm reproduction and small project practice. The goal is not only to re-implement classic models, but also to record ideas, test assumptions, and turn those ideas into working code. The project is updated continuously as new concepts are explored.

## Overview

The codebase contains several self-contained examples covering regression, classification, multi-label text tagging, and a small neural network regression task. Most scripts generate synthetic data or use a public dataset, train a simple model, and visualize the results so the learning process is easy to inspect.

## Repository Structure

- `Linear_Regression.py` - A from-scratch linear regression example using gradient descent on simulated data.
- `Logistic_Regression.py` - A simple logistic regression-style experiment with sigmoid activation and binary-style optimization.
- `Traffic_Flow_Prediction.py` - A neural network regression demo that predicts traffic volume from synthetic features.
- `Wine_Quality.py` - A PyTorch binary classification example built on the UCI wine quality dataset.
- `auto_stack.py` - A multi-label text tagging demo using TF-IDF features and a neural network classifier.

## What This Project Focuses On

- Reproducing core machine learning ideas in a clear and compact way.
- Understanding the full workflow from data preparation to training and evaluation.
- Recording insights gained during study and turning them into executable experiments.
- Iterating on basic models before moving to larger or more realistic projects.

## Requirements

The examples rely on a Python environment with the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`

Some scripts also download data from the internet, so an active network connection may be required.

## Installation

If you are using a virtual environment, activate it first, then install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

If you already have a project-specific environment, make sure it is selected before running any script.

## Usage

Each file can be run independently:

```bash
python Linear_Regression.py
python Logistic_Regression.py
python Traffic_Flow_Prediction.py
python Wine_Quality.py
python auto_stack.py
```

Most scripts will print training progress to the console and display plots for model behavior, loss curves, or prediction results.

## Script Details

### Linear Regression

This script creates synthetic data, initializes random parameters, and uses gradient descent to fit a straight line. It also plots the training loss and the final fitted line.

### Logistic Regression

This example demonstrates a sigmoid-based binary classification workflow on simulated data. It includes loss calculation, gradient updates, and visualization of the learned decision trend.

### Traffic Flow Prediction

This project generates a synthetic traffic dataset from time, day, and weather features, then trains a small PyTorch regression model to predict traffic volume. The trained model is saved as `traffic_prediction_model.pth`.

### Wine Quality

This script loads the UCI red wine quality dataset, converts the target into a binary label, standardizes the features, and trains a feed-forward neural network for classification.

### Auto Stack

This example builds a tiny multi-label text classification pipeline. It vectorizes text with TF-IDF, encodes labels with `MultiLabelBinarizer`, trains a neural network, and predicts tags for a new question.

## Notes

- The repository is designed for study and experimentation, so the scripts are intentionally compact and easy to modify.
- Some examples use synthetic data for clarity rather than realism.
- The naming and implementation style may evolve as new machine learning ideas are added.

## Future Direction

Potential next steps for the repository include:

- Adding more classical machine learning algorithms.
- Improving data preprocessing and evaluation pipelines.
- Organizing examples by topic or difficulty.
- Adding reusable utilities for training, plotting, and metrics.

## License

No license has been specified yet. Add one if you plan to share or reuse the code publicly.
