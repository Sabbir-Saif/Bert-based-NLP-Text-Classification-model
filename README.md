# BERT Phishing Site Classifier
This repository contains a finetuned Large Language Model(LLM) built using the BERT architecture to classify phishing websites. 
The model is trained on the "shawhin/phishing-site-classification" dataset and leverages the `transformers` library by Hugging Face. 
It classifies websites as either "Safe" or "Not Safe" with evaluation metrics including accuracy and AUC (Area Under the Curve).

## Overview

- **Model**: `google-bert/bert-base-uncased` fine-tuned for binary classification.
- **Dataset**: `shawhin/phishing-site-classification` from Hugging Face Datasets.
- **Task**: Binary classification of phishing websites.
- **Metrics**: Accuracy and ROC-AUC.
- **Framework**: PyTorch via Hugging Face Transformers.

## Features

- Utilizes BERT's pre-trained weights with a custom classification head.
- Freezes all layers except the pooler for efficient fine-tuning.
- Supports training and evaluation with configurable hyperparameters.
- Includes logging and model checkpointing.

## Prerequisites

- **Python 3.8+**
- **Required Libraries**:
  - `transformers`
  - `datasets`
  - `evaluate`
  - `numpy`
  - `torch`

## Hyperparameters:
Learning Rate: 2e-4
Batch Size: 8
Epochs: 10
Output Directory: bert-phishing-classifier_teacher
These can be adjusted in the training_args section of the script.
