GoEmotions Emotion Classification: TF-IDF vs BiLSTM vs DistilBERT
A controlled comparison of a classical baseline, an RNN, and a fine-tuned transformer on 28-class emotion classification of Reddit comments (GoEmotions), with per-class and error analysis.
Key result: DistilBERT reaches 0.478 macro-F1 vs 0.421 for TF-IDF + Logistic Regression and 0.428 for a BiLSTM. The gain is [largest / smallest] on [rare / frequent] classes. Most remaining errors are between semantically close emotions (e.g. [pair 1], [pair 2]), which suggests label ambiguity as much as model weakness.
