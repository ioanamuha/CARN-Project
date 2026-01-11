# 📚 Context-Aware Word Sense Disambiguation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Advanced Topics in Neural Networks (ATNN) - Assignment 4**
>
> A Hybrid Deep Learning approach for **SemEval 2026 Task 5**: *Rating Plausibility of Word Senses in Ambiguous Sentences*.

---

## 🧠 About The Project

Word Sense Disambiguation (WSD) is a core challenge in NLP, especially in creative writing where context shifts the meaning of words subtly. This project tackles the problem of **Polysemy** in short stories using the **AmbiStory** dataset.

The goal is to predict a continuous plausibility score (1.0 - 5.0) for a specific definition of a homonym (e.g., *"bank"*) given a 5-sentence story.

### **The Solution: Cross-Encoder BERT-LSTM**
We moved beyond static embeddings (GloVe) to a **Hybrid Architecture** that combines:
1.  **Cross-Encoder Input:** Concatenating Definition + Story to enable early attention interaction.
2.  **BERT (Frozen):** For deep contextual feature extraction.
3.  **Projection Bottleneck:** Reducing dimensionality to prevent overfitting on small data.
4.  **Bi-LSTM:** For modeling the narrative sequence and dependencies.

---

## 📊 Performance & Ablation Study

We benchmarked several architectures to isolate the impact of contextual embeddings.

| Architecture | Embedding | Strategy | Spearman ($r$) | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline LSTM** | GloVe | Siamese | `~ -0.01` | 52.5% | ❌ Failed |
| **Siamese BiLSTM** | GloVe | Attention | `0.06` | 53.5% | ⚠️ Overfit |
| **Hybrid Siamese** | BERT | Separate | `0.10` | 57.3% | 📈 Better |
| **Cross-Encoder** | **BERT** | **Concatenated** | **0.15** | **59.0%** | ✅ **Best** |

> *Accuracy is measured as the percentage of predictions within 1 Standard Deviation of the human average.*

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YourUsername/CARN-project.git]
    cd CARN-project
    ```

2.  **Install dependencies**
    ```bash
    pip install torch transformers scipy numpy
    ```

3.  **Data Setup**
    Ensure your directory structure looks like this:
    ```text
    data/
    ├── train.json
    ├── dev.json
    └── ref/
        └── solution.jsonl  (For evaluation)
    ```

---

## 🚀 Usage

### Training the Model
To train the Cross-Encoder BERT-LSTM model, simply run:

```bash
python main.py
```

This will:
1.  Load the **BERT-base-uncased** tokenizer.
2.  Train the model for **25 epochs** (with early stopping/scheduling).
3.  Save the best model to `best_bert_lstm.pth`.
4.  Generate predictions in `predictions/bert_lstm_predictions.jsonl`.
5.  Automatically run the official scoring script.

### Evaluation Only
If you have a saved model and want to evaluate it:

```bash
python scoring.py data/ref/solution.jsonl predictions/bert_lstm_predictions.jsonl output/scores.json
```

## 🔗 References
- SemEval Task: Rating Plausibility of Word Senses
- Base Paper: State-of-the-Art Approaches to Word Sense Disambiguation
- Dataset: AmbiStory