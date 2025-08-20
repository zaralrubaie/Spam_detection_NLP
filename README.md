# Spam Detection using NLP 📩🤖

This project builds a model to classify messages as **"spam"** or **"ham"** (non-spam) using **Natural Language Processing (NLP)**.  
The goal is to automatically detect spam messages, which is useful for **email systems, text messaging services, and other communication platforms**.

---

## 🔹 Model Evaluation

The model was evaluated on a test set, and the performance metrics are summarized below:

| Class / Metric   | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| 0 (Ham)         | 0.98      | 0.99   | 0.99     | 965     |
| 1 (Spam)        | 0.96      | 0.89   | 0.93     | 150     |
| **Accuracy**    | -         | -      | 0.98     | 1115    |
| **Macro Avg**   | 0.97      | 0.94   | 0.96     | 1115    |
| **Weighted Avg**| 0.98      | 0.98   | 0.98     | 1115    |

---

## 🔹 Evaluation Metrics Explained

- **Accuracy**: The percentage of correct predictions out of all predictions.  

- **Precision**: Measures how many of the predicted spam messages were actually spam.  

\[
 \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
\]

- **Recall (Sensitivity)**: Measures how many actual spam messages were correctly identified by the model.  

\[
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
\]

- **F1-Score**: Harmonic mean of precision and recall, balancing the two metrics.  

\[
  F1\text{-}score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}
\]

> These metrics help evaluate the model's ability to correctly classify spam and ham messages, balancing both **false positives** and **false negatives**.

---

## 🔹 Notes

- The **classification report** provides a summary of these metrics.  
- The **confusion matrix** helps visualize performance across different classes (spam vs ham).  
- This project demonstrates the practical application of NLP for **text classification** tasks.
