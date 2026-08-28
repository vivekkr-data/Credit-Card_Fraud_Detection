# Credit Card Fraud Detection: Viva Answers

## Questions and Answers

### 1. What is credit card fraud detection?

It is the process of using transaction data to identify whether a credit card transaction is normal or fraudulent.

### 2. What is the target variable?

The target variable is `Class`.

### 3. What do Class 0 and Class 1 mean?

Class 0 means a normal transaction. Class 1 means a fraudulent transaction.

### 4. Why is this dataset imbalanced?

Fraud is rare. The original data has 492 fraud transactions but 284,315 normal transactions.

### 5. How many fraud samples are in the original dataset?

There are 492 fraud samples in the original dataset. After removing exact duplicates, 473 remain.

### 6. What are V1-V28?

They are anonymized numerical features created using PCA. Their original business meanings are not available.

### 7. What are Time and Amount?

`Time` records elapsed transaction time, and `Amount` records the transaction value.

### 8. Why were duplicate rows removed?

Duplicate rows could place identical records in both training and evaluation data, making the results less reliable.

### 9. What is data leakage?

Data leakage happens when information from validation or test data influences model training and produces unrealistically good results.

### 10. Why do we split data before SMOTE?

Splitting first prevents synthetic samples from using information from validation or test records.

### 11. What is SMOTE?

SMOTE is an oversampling method that creates synthetic examples of the minority class.

### 12. How does SMOTE work at a basic level?

It selects nearby minority-class samples and creates new points between them.

### 13. Why is SMOTE applied only to training data?

Validation and test data must keep the real class distribution so evaluation remains fair.

### 14. Why do we use a validation set?

We use it to compare Random Forest and XGBoost without touching the final test set.

### 15. Why is the final test set kept untouched?

It gives an unbiased final estimate of how the selected model performs on unseen data.

### 16. What is StandardScaler?

StandardScaler transforms a feature to have approximately zero mean and unit standard deviation.

### 17. Why are Time and Amount scaled?

They have different ranges, and scaling helps SMOTE calculate distances more fairly. V1-V28 are already PCA-transformed.

### 18. What is Random Forest?

Random Forest combines many decision trees and uses their combined prediction.

### 19. What is XGBoost?

XGBoost is a boosting algorithm that builds trees in sequence, with later trees learning from earlier errors.

### 20. What is the difference between Random Forest and XGBoost?

Random Forest builds mostly independent trees and combines them. XGBoost builds trees sequentially to correct previous mistakes.

### 21. What is Accuracy?

Accuracy is the number of correct predictions divided by all predictions.

### 22. Why is Accuracy misleading here?

A model can predict almost everything as normal and still get very high Accuracy because fraud is extremely rare.

### 23. What is Precision?

Precision tells us how many transactions predicted as fraud were actually fraud.

### 24. What is Recall?

Recall tells us how many actual fraud transactions the model detected.

### 25. What is F1 Score?

F1 Score is the harmonic mean of Precision and Recall. It balances both metrics.

### 26. What is ROC-AUC?

ROC-AUC measures how well the model separates fraud and normal classes across different thresholds.

### 27. What is a False Positive?

It is a normal transaction incorrectly predicted as fraud.

### 28. What is a False Negative?

It is a fraudulent transaction incorrectly predicted as normal.

### 29. Why are False Negatives important in fraud detection?

They represent missed fraud, which may cause financial loss and security risk.

### 30. How did you select the final model?

I compared both models on the same validation set and used F1 Score as the main metric. Random Forest won with validation F1 of 0.797546.

### 31. What does Joblib do?

Joblib saves and loads Python machine-learning objects such as the trained model and scaler.

### 32. Why save the scaler?

New transactions must receive exactly the same Time and Amount transformation used during training.

### 33. Why save feature names?

They preserve the exact input columns and order expected by the model.

### 34. What is Streamlit?

Streamlit is a Python framework for building interactive data and machine-learning web applications.

### 35. How does Streamlit make a prediction?

It validates the inputs, orders the features, transforms Time and Amount with the saved scaler, and calls the saved model.

### 36. Why is creditcard.csv not uploaded to GitHub?

It is about 150 MB, which is above GitHub's normal 100 MB single-file limit.

### 37. Does the deployed Streamlit app need the training dataset?

No. It needs only the saved model, scaler, feature names, and model information.

### 38. What are the limitations of this project?

The features are anonymized, the data is historical, fraud patterns can change, and the default threshold is not optimized for a bank's real costs.

### 39. How can the project be improved?

It can use newer data, threshold tuning, model monitoring, periodic retraining, and cost-based evaluation.

### 40. Explain the complete project in 60 seconds.

I built a credit card fraud detection system using the provided dataset. I checked the data and removed 1,081 exact duplicates before creating stratified train, validation, and final test sets. I fitted StandardScaler only on training data for Time and Amount, then applied SMOTE only to training data. I trained Random Forest and XGBoost and compared them on Precision, Recall, F1 Score, and ROC-AUC. Random Forest had the better validation F1, so I trained a fresh Random Forest on combined training and validation data and evaluated it once on the untouched test set. It achieved 0.728155 F1 and 0.975566 ROC-AUC. I saved the model and preprocessing objects with Joblib and built a Streamlit app for single and batch predictions.

## Resume Bullets

- Built an end-to-end Credit Card Fraud Detection system for highly imbalanced transaction data, removing 1,081 exact duplicates and using leakage-safe stratified train, validation, and final test splits.
- Applied StandardScaler and SMOTE only to development data, then compared Random Forest and XGBoost with Precision, Recall, F1, and ROC-AUC; selected Random Forest using a 0.797546 validation F1 Score.
- Developed a Joblib-powered Streamlit application for single and batch fraud predictions; the untouched test set achieved 0.675676 Precision, 0.789474 Recall, 0.728155 F1, and 0.975566 ROC-AUC.
