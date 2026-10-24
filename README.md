# Transcription Factor Binding Site Prediction in DNA Sequences using CNNs and BiLSTMs

A deep learning pipeline that predicts **Transcription Factor Binding Sites (TFBS)** across **690 sparse ChIP-seq targets** from 1,000 bp DNA sequences, using a CNN + BiLSTM architecture inspired by DanQ (Quang & Xie, 2016).

## Attribution
Based on code from *Deep Learning for Genomics* by Upendra Kumar Devisetty (Packt, 2022). The book describes 4,863,024 sequences split into 4,400,000 training, 8,000 validation, and 455,024 test samples, which matches the DeepSEA benchmark (Zhou & Troyanskaya, 2015). 

## Data
This project uses a subset of that dataset:
* 10,000 training and 1,000 test sequences
* Each sequence is a 1000 × 4 one-hot matrix (A, C, G, T)
* 690 binary labels per sequence, with about 1.25% positives

## Modifications to the Original
Starting from the book's baseline CNN + BiLSTM model, I made the following changes:

1. **Replaced `Flatten()` with `GlobalMaxPooling1D()`**. Flattening the (487, 640) BiLSTM output gave the 695-unit dense layer 311,680 inputs and ~216.6M parameters on its own. Global max pooling keeps the strongest activation per feature, so that layer now has 640 inputs and ~445K parameters.
2. **Added reverse-complement (RC) augmentation.** DNA is double-stranded, so a sequence and its reverse complement represent the same locus, and TF binding is functionally equivalent on both strands for most interactions. Training sequences are flipped along both the position and channel axes, with labels unchanged.
3. **Computed the positive label fraction (~1.25%).** Because the targets are sparse, accuracy is misleading (predicting all zeros scores ~98.75%), which motivated the choice of metrics below.
4. **Added training callbacks.** `EarlyStopping` and a learning-rate scheduler (`ReduceLROnPlateau`), and increased the maximum number of epochs from 5 to 20.
5. **Evaluated with threshold-independent metrics.** Target-wise mean ROC-AUC and mean PR-AUC, instead of accuracy.

---

## Model Architecture
```text
Input (1000 x 4 One-Hot Matrix)
 └── Conv1D (320 filters, kernel size 26, ReLU) → (975, 320)
      └── MaxPooling1D (pool size 2) → (487, 320)
           └── Dropout (0.1)
                └── Bidirectional LSTM (320 units per direction) → (487, 640)
                     └── Dropout (0.5)
                          └── GlobalMaxPooling1D → (640)
                               └── Dense (695) + ReLU
                                    └── Dense (690) + Sigmoid → 690 targets
```

| Model | Pooling/flatten layer | Dense (695) params | Total params |
|---|---|---|---|
| Original | `Flatten` → 311,680 inputs | 216,618,295 | 218,773,095 (~835 MB) |
| Modified | `GlobalMaxPooling1D` → 640 inputs | 445,495 | 2,600,295 (~10 MB) |

## Results
| Metric | Value |
|---|---|
| Mean ROC-AUC | 0.668 |
| Mean PR-AUC | 0.037 |

These come from a small-scale run (10,000 training sequences), so they are a baseline and not comparable to published DeepSEA/DanQ results.