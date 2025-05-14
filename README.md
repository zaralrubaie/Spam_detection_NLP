#spam detection using NLP

This project aims to build a model that can classify messages as "spam" or "ham" (non-spam) using Natural Language Processing (NLP).
The goal is to automatically detect spam messages, which is useful for email systems, text messaging services, and other communication platforms.

model eval metrix was : 
 precision    recall  f1-score   support

           0       0.98      0.99      0.99       965
           1       0.96      0.89      0.93       150

    accuracy                           0.98      1115
   macro avg       0.97      0.94      0.96      1115
weighted avg       0.98      0.98      0.98      1115
## Evaluation Metric

The performance of the spam detection model is evaluated using the following metrics:

- **Accuracy**: The percentage of correct predictions out of the total predictions made.
  
- **Precision**: Precision measures the accuracy of positive predictions. It tells us how many of the predicted spam messages were actually spam.

  \[
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
  \]

- **Recall (Sensitivity)**: Recall measures how many actual spam messages were correctly identified by the model.

  \[
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
  \]

- **F1-Score**: The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.

  \[
  F1\text{-}score = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision + Recall}}
  \]

These metrics help in evaluating the model's ability to correctly classify spam and ham messages, balancing both false positives and false negatives.

The classification report printed after running the model gives a summary of these metrics, and the confusion matrix helps visualize the performance of the classifier across different classes (spam vs ham).



