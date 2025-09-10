# Spam Detection using NLP 📩🤖

This project builds a model to classify messages as spam or ham (non-spam) using Natural Language Processing (NLP).
The goal is to automatically detect unwanted messages, which is useful for email systems, messaging apps, and other communication platforms.

## 🔹Model Performance

The model was evaluated on a test set, and the main results are summarized below:

| Class / Metric   | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| 0 (Ham)         | 0.98      | 0.99   | 0.99     | 965     |
| 1 (Spam)        | 0.96      | 0.89   | 0.93     | 150     |
| **Accuracy**    | -         | -      | 0.98     | 1115    |
| **Macro Avg**   | 0.97      | 0.94   | 0.96     | 1115    |
| **Weighted Avg**| 0.98      | 0.98   | 0.98     | 1115    |
- Interpretation: The model is very accurate at identifying both spam and ham messages, with slightly lower recall for spam.
---

## 🔹 What the Metrics Mean
- Accuracy: Percentage of all messages that were classified correctly.
- Precision: Of all messages predicted as spam, how many were actually spam.
- Recall: Of all actual spam messages, how many did the model correctly identify.
- F1-Score: A balance between precision and recall, giving a single performance score.

- These metrics show how well the model can separate spam from legitimate messages, while minimizing false positives (ham marked as spam) and false negatives (spam missed).

---

## 🔹 Key Takeaways
- The model performs very well for real-world spam detection.
- Confusion matrix can be used to visualize the predictions.
- Demonstrates the practical application of NLP in text classification tasks.
---
## 🔹 Project Structure
````
spam-detection-nlp/
│
├── spam_detector_model.py     # Python script with preprocessing and model training
├── predictions.csv            # CSV file with predicted results
└── README.md                  # Project documentation
````
## 🔹 License
This project is licensed under the MIT License 
