# Emotion Classification on GoEmotions

A comparative study of three modeling approaches for fine-grained emotion classification on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset (27 emotions + neutral = **28 classes**). The project benchmarks a classical linear baseline, a recurrent network trained from scratch, and a fine-tuned pretrained transformer under one shared, imbalance-aware evaluation protocol.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Usage](#usage)

## Overview

Emotion classification is a hard multi-class problem: categories overlap semantically, labels are subjective, and the class distribution is highly skewed. This project compares three approaches of increasing complexity:

| # | Approach | Type |
|---|----------|------|
| 1 | TF-IDF + Logistic Regression | Classical ML baseline |
| 2 | BiLSTM | Neural network trained from scratch |
| 3 | DistilBERT | Fine-tuned pretrained transformer |

Because of the class imbalance, **macro-F1** is the primary metric. Accuracy and weighted-F1 are reported for completeness. All three models use class-weighted training and are evaluated on the same held-out test set.

## Dataset

[GoEmotions](https://arxiv.org/abs/2005.00547) is a corpus of English Reddit comments annotated with 27 emotion categories plus neutral. The notebook loads the `simplified` configuration from the Hugging Face Hub (`google-research-datasets/go_emotions`).

**Preprocessing:** only comments with exactly **one** label are kept, turning the task into single-label, 28-way classification.

| Split | Original rows | Single-label rows used |
|-------|:-------------:|:----------------------:|
| Train | 43,410 | 36,308 |
| Validation | 5,426 | 4,548 |
| Test | 5,427 | 4,590 |

The training set is **heavily imbalanced**: `neutral` has roughly 12,800 examples, while classes such as `grief`, `pride`, `relief`, and `nervousness` have only a handful. To counter this, every model is trained with class weights computed using scikit-learn's `"balanced"` strategy (inverse class frequency).

![Class distribution](figures/class_distribution.png)

## Models

### 1. TF-IDF + Logistic Regression
- `TfidfVectorizer` with word unigrams and bigrams (`ngram_range=(1, 2)`), `min_df=2`, `sublinear_tf=True`
- `LogisticRegression` with `class_weight="balanced"` and `max_iter=2000`
- Regularization strength `C` selected from `{1, 3, 10}` by validation macro-F1 (best: `C=10`)

### 2. BiLSTM (trained from scratch)
- Regex tokenizer (lowercased words, plus `! ? . ,`) and a vocabulary built from the training set; tokens with frequency ≥ 2 are kept (vocabulary size **12,001**, including `<pad>` and `<unk>`)
- Sequences truncated to 50 tokens; batch size 64
- Architecture: embedding (128-d) → dropout (0.4) → bidirectional LSTM (hidden size 128 per direction) → masked **max-pooling** over time → dropout → linear classifier
- Class-weighted cross-entropy, Adam (`lr=2e-3`), gradient clipping (max norm 1.0)
- 12 epochs; the checkpoint with the best validation macro-F1 is restored for testing

### 3. DistilBERT
- `distilbert-base-uncased` fine-tuned for sequence classification with the Hugging Face `Trainer`
- Custom `WeightedTrainer` that applies class-weighted cross-entropy
- Tokenization truncated at 64 tokens with dynamic padding (`DataCollatorWithPadding`)
- 4 epochs, batch size 32, learning rate `5e-5`, weight decay 0.01, 10% linear warmup, mixed precision (fp16) on GPU
- Best checkpoint selected by validation macro-F1 (`load_best_model_at_end=True`)

## Results

All numbers are on the held-out **test set** (4,590 examples).

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|:--------:|:--------:|:-----------:|
| TF-IDF + LogReg | **0.506** | 0.421 | **0.515** |
| DistilBERT | 0.492 | **0.478** | 0.485 |
| BiLSTM | 0.456 | 0.428 | 0.458 |

Aggregate metrics are also saved to [`results.csv`](results.csv).

**Key takeaways**

- **DistilBERT has the best macro-F1 (0.478)**, meaning the most balanced performance across all 28 classes, including rarer emotions.
- **TF-IDF + LogReg has the highest accuracy (0.506) and weighted-F1 (0.515)**. These metrics are dominated by frequent classes, especially `neutral` (1,606 of 4,590 test examples), where the linear model scores much higher (F1 0.566 vs. 0.371 for DistilBERT and 0.392 for BiLSTM). It remains a strong, fast baseline.
- **BiLSTM** edges out the TF-IDF baseline on macro-F1 (0.428 vs. 0.421) but trails both other models on accuracy and weighted-F1.
- Across models, classes with distinctive lexical cues (`gratitude`, `love`, `amusement`) score highest, while subtle or rare classes (`relief`, `realization`, `grief`) score lowest.

### Validation tuning

**TF-IDF + LogReg:** validation macro-F1 by regularization strength

| C | Val macro-F1 |
|:-:|:------------:|
| 1 | 0.433 |
| 3 | 0.440 |
| 10 | **0.445** |

**BiLSTM:** validation macro-F1 rose from 0.269 (epoch 1) to a peak of **0.444 at epoch 11**, while training loss fell from 2.951 to 1.466 over 12 epochs. The epoch-11 checkpoint was used for testing.

### Per-class F1

Test F1 for every class and model, sorted by DistilBERT performance:

![Per-class F1](figures/per_class_f1.png)


## Usage

1. Launch Jupyter (or open the notebook in VS Code):
   ```bash
   jupyter notebook
   ```
2. Open `geoemotions_classifications.ipynb` and select a kernel with the required packages (`datasets`, `transformers`, `accelerate`, `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`).
3. Run all cells. The notebook will:
   - Download and filter the GoEmotions dataset
   - Plot the class distribution
   - Train and evaluate TF-IDF + LogReg, DistilBERT, and BiLSTM
   - Save figures to `figures/` and aggregate metrics to `results.csv`

The `figures/` directory is created automatically by the notebook. The first cell (`!pip -q install datasets transformers accelerate`) installs the Hugging Face packages if you run in a fresh environment such as Colab.
