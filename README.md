# Speech Emotion Recognition Using 1D CNNs

A deep learning project for **speech emotion recognition (SER)** using MFCC-based audio features and 1D Convolutional Neural Networks.

The project uses the **IEMOCAP dataset** to classify speech into four emotion categories: Angry, Happy, Neutral, and Sad.

Two neural network architectures are implemented and evaluated:

- Plain 1D CNN
- Residual 1D CNN

The project explores audio feature extraction, speaker-independent evaluation, neural network design, and comparative model evaluation using PyTorch.

---

## Overview

Speech emotion recognition aims to identify emotional states from speech signals.

Human speech contains acoustic information that can help distinguish emotional expressions, including differences in spectral characteristics, pitch-related patterns, and temporal dynamics.

This project implements a speech emotion classification pipeline that converts audio recordings into numerical features and uses deep learning models to predict the associated emotion.

The four target classes are:

- Angry
- Happy
- Neutral
- Sad

The main objective is to compare a conventional 1D CNN with a residual CNN for classifying speech emotion from acoustic features.

## Key Features

- IEMOCAP dataset integration
- Four-class speech emotion classification
- MFCC feature extraction
- Optional Delta and Delta-Delta features
- Audio sequence padding and truncation
- Feature standardization
- Plain 1D CNN architecture
- Residual 1D CNN architecture
- Speaker-independent and random data splitting
- Model checkpointing based on validation loss
- Accuracy and F1-score evaluation
- Confusion matrix visualization
- Training and validation loss curves
- Audio duration and class distribution analysis

---

## Dataset

This project uses the **IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset**, loaded through the Hugging Face Datasets library.

Dataset identifier:

`AbstractTTS/IEMOCAP`

IEMOCAP is an acted, multimodal emotional interaction dataset containing speech and associated annotations. It is widely used in research on speech emotion recognition and affective computing.

In this implementation, only four emotion categories are selected:

| Emotion | Label ID |
|---|---:|
| Angry | 0 |
| Happy | 1 |
| Neutral | 2 |
| Sad | 3 |

Samples with other emotion labels are excluded from the classification task.

The dataset's original speaker information is also extracted to support speaker-independent evaluation.

**Dataset access and licensing:** IEMOCAP is a research dataset with access conditions determined by its providers. Users should consult the dataset documentation and applicable terms before downloading, redistributing, or publishing its audio recordings.

---

## Project Pipeline

The complete workflow is organized into the following stages:

```text
IEMOCAP Dataset
       |
       v
Emotion Selection
       |
       v
Audio Feature Extraction
       |
       v
MFCC / Delta / Delta-Delta
       |
       v
Sequence Padding or Truncation
       |
       v
Feature Standardization
       |
       v
Train / Validation / Test Split
       |
       v
+---------------------------+
|                           |
v                           v
Plain 1D CNN          Residual 1D CNN
|                           |
+-------------+-------------+
              |
              v
       Model Evaluation
              |
              v
 Accuracy / F1 / Confusion Matrix
```

---

## Audio Feature Extraction

Raw audio waveforms are converted into features using **Mel-Frequency Cepstral Coefficients (MFCCs)**.

MFCCs represent important aspects of the short-term spectral envelope of an audio signal and are commonly used in speech processing tasks.

The project uses Librosa to extract 40 MFCC coefficients.

### MFCC Features

The main configuration is:

| Parameter | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Number of MFCC coefficients | 40 |
| Hop length | 512 samples |
| Maximum sequence length | 300 frames |

### MFCC + Delta + Delta-Delta

The project supports two feature configurations.

**MFCC only**

Uses 40 coefficients per audio frame.

**MFCC + Delta + Delta-Delta**

Combines:

- MFCC: spectral features
- Delta: first-order temporal differences
- Delta-Delta: second-order temporal differences

With 40 MFCC coefficients, the combined representation contains 120 feature channels.

The additional features provide information about how the acoustic characteristics change over time.

The configuration is controlled by:

```python
USE_DELTA = True
```

When enabled, the model receives 120 input channels instead of 40.

---

## Audio Sequence Standardization

Speech recordings have different durations and therefore produce feature sequences with different numbers of frames.

To create batches of consistent shape, each feature sequence is padded or truncated to a maximum length of 300 frames.

```python
MAX_LEN = 300
KEEP = "last"
```

Two truncation strategies are supported:

- `first`: retain the beginning of the sequence.
- `last`: retain the end of the sequence.

Shorter sequences are padded with zeros.

This produces fixed-size input tensors suitable for batch processing.

---

## Feature Normalization

The extracted features are standardized using the training set's mean and standard deviation.

The normalization statistics are calculated from the training features and then applied to the validation and test sets.

This avoids calculating separate normalization statistics from the evaluation data.

---

## Data Splitting

The implementation supports two data splitting strategies.

### Random Split

The random split uses stratified sampling to preserve the overall emotion distribution across the dataset partitions.

The approximate proportions are:

| Split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 10% |
| Test | 20% |

Stratification is performed using emotion labels.

### Speaker-Independent Split

The project also supports splitting the dataset according to speaker identity.

The speaker-independent configuration assigns different speakers to training, validation, and test sets.

This approach is important because speech characteristics can vary substantially between individuals.

A model evaluated on speakers it has not seen during training provides a more meaningful test of generalization to new speakers.

Available options:

- `"random"`
- `"speaker"`

To evaluate generalization to unseen speakers, use the speaker-independent configuration and report its results separately from random splitting.

---

## Model Architectures

Two 1D CNN architectures are implemented using PyTorch.

### 1. Plain 1D CNN

The Plain CNN is the baseline architecture.

It consists of three convolutional blocks followed by global average pooling and a fully connected classifier.

Each convolutional block contains:

- 1D convolution
- Batch normalization
- ReLU activation
- Max pooling

Architecture:

```text
Input Features
      |
      v
ConvBlock: 64 Channels
      |
      v
ConvBlock: 128 Channels
      |
      v
ConvBlock: 256 Channels
      |
      v
Adaptive Average Pooling
      |
      v
Fully Connected: 128
      |
      v
Output: 4 Classes
```

This architecture learns hierarchical temporal representations from the acoustic features.

### 2. Residual 1D CNN

The Residual CNN uses residual blocks to learn feature representations through convolutional layers and shortcut connections.

Each residual block contains two convolutional layers, batch normalization, ReLU activation, and a shortcut connection.

When the input and output channel dimensions differ, a 1D convolution is used to match the dimensions of the shortcut.

Architecture:

```text
Input Features
      |
      v
ResidualBlock: 64 Channels
      |
      v
ResidualBlock: 128 Channels
      |
      v
ResidualBlock: 256 Channels
      |
      v
Adaptive Average Pooling
      |
      v
Fully Connected: 128
      |
      v
Output: 4 Classes
```

Residual connections provide an alternative path for information and gradients to flow through the network.

This architecture is compared with the Plain CNN to investigate whether residual connections improve classification performance.

---

## Training Configuration

The principal training parameters are:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Number of epochs | 30 |
| Number of classes | 4 |
| Optimizer | Adam |
| Loss function | Cross-Entropy Loss |
| Device | CUDA if available, otherwise CPU |

The best model checkpoint is selected using validation loss.

This helps preserve the model parameters from the epoch with the lowest validation loss rather than automatically using the final epoch.

---

## Evaluation Metrics

Both architectures are evaluated on the test set.

The project calculates three primary metrics.

### Accuracy

The proportion of correctly classified test samples.

### Weighted F1-Score

The F1-score for each class is combined using the number of true samples in that class as its weight.

This metric accounts for class imbalance by giving more influence to classes with more examples.

### Macro F1-Score

The F1-score is calculated independently for each class and then averaged equally across all classes.

Macro F1 is useful for assessing performance across all emotion categories, including less frequent classes.

### Confusion Matrix

A confusion matrix visualizes the relationship between true emotion labels and predicted labels.

It helps identify which emotions the model confuses most frequently.

The project also plots training and validation loss curves to examine learning behavior and potential overfitting.

---

## Model Comparison

The experiment compares the following architectures:

- Plain 1D CNN
- Residual 1D CNN

Use the results generated by the notebook to complete the table.

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Plain 1D CNN | — | — | — |
| Residual 1D CNN | — | — | — |

The best-performing architecture should be determined from the actual evaluation results rather than assumed in advance.

For a fair comparison, both models should use the same data split, feature configuration, training budget, and evaluation procedure.

---

## Visualizations

The notebook generates several visualizations to support dataset analysis and model evaluation.

### Emotion Distribution

A bar chart displays the number of samples in each emotion category.

This helps identify class imbalance in the selected dataset.

### Audio Duration Distribution

A histogram shows the distribution of audio recording durations.

Additional statistics include:

- Mean duration
- Standard deviation
- Minimum duration
- Maximum duration

### Training and Validation Loss

Loss curves show how the training and validation losses change across epochs.

They can help identify convergence behavior and potential overfitting.

### Confusion Matrix

The confusion matrix displays the number of correctly and incorrectly classified samples for each emotion.


## Installation

Install the required dependencies:

```bash
pip install torch datasets librosa numpy pandas matplotlib scikit-learn
```

The project uses the Hugging Face Datasets library to load the selected dataset.

Depending on the environment, downloading the dataset may require additional setup or access permissions.

---

## Running the Project

1. Install the dependencies.
2. Open `speech_emotion_recognition.ipynb`.
3. Load and filter the IEMOCAP dataset.
4. Select the splitting strategy.
5. Configure the audio features.
6. Train and evaluate the Plain CNN and Residual CNN.
7. Compare the evaluation metrics and confusion matrices.
8. Update the model comparison table with the final test results.

### Configuration Options

The notebook allows experiments with different settings:

```python
SPLIT_MODE = "random"
USE_DELTA = True
KEEP = "last"
MODEL_NAME = "plain"
```

The model comparison section explicitly trains both architectures, so `MODEL_NAME` does not determine which models are included in that comparison loop.

---

e can be distributed under a license selected by the repository owner. Dataset access, redistribution, and usage remain subject to the IEMOCAP dataset's applicable terms.
