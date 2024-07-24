# Online Payment Fraud Detection

## Introduction

Online payment is the most popular transaction method in the world today. However, with an increase in online payments comes a rise in payment fraud. The objective of this study is to identify fraudulent and non-fraudulent payments using a dataset collected from Kaggle, which contains historical information about fraudulent transactions. This dataset can be used to detect fraud in online payments.

## Dataset

The dataset consists of 10 variables and 6,362,620 observations:

- `step`: Represents a unit of time where 1 step equals 1 hour
- `type`: Type of online transaction
- `amount`: The amount of the transaction
- `nameOrig`: Customer starting the transaction
- `oldbalanceOrg`: Balance before the transaction
- `newbalanceOrig`: Balance after the transaction
- `nameDest`: Recipient of the transaction
- `oldbalanceDest`: Initial balance of recipient before the transaction
- `newbalanceDest`: The new balance of recipient after the transaction
- `isFraud`: Fraud transaction indicator (0 for non-fraudulent, 1 for fraudulent)

## Exploratory Data Analysis

The data exploration phase involves checking for missing values, data types, and duplicate values. Visualizations are used to understand the distribution of transaction types, amounts, and the balance amounts before and after transactions for both the origin and destination accounts. Key observations include:

- Cash out is the most common transaction type.
- The distribution of transaction amounts is right-skewed.
- There are more non-fraudulent transactions compared to fraudulent transactions.
- Fraudulent transactions occur only in transfer and debit types and are associated with very low transaction amounts.

## Data Preprocessing

To prepare the data for modeling:

1. **Downcast numerical columns**: This helps to reduce memory usage.
2. **Convert categorical columns to category dtype**: This optimizes memory usage and processing time.
3. **Remove irrelevant columns**: Columns such as `nameOrig`, `nameDest`, and balance columns are dropped to simplify the dataset.
4. **Handle class imbalance**: `RandomUnderSampler` is used to balance the classes in the training dataset.

## Model Building

Two models are used to identify online payment fraud:

1. **Random Forest Classifier**
2. **Logistic Regression**

### Cross-Validation

Stratified K-Fold cross-validation is used to evaluate the models. Performance metrics include accuracy, precision, recall, F1 score, and ROC-AUC score. The Random Forest Classifier outperforms Logistic Regression in all metrics.

### Model Evaluation

The Random Forest model is further evaluated using a confusion matrix and ROC curve on the test set. The model achieves an AUC score of 0.999, indicating excellent performance in distinguishing between fraudulent and non-fraudulent transactions.

## Conclusion

Random Forest Classifier is identified as the best performing model for detecting fraudulent online payments, achieving an AUC of 0.999. This indicates a high capability of the model to distinguish between fraud and non-fraud transactions.

## Installation and Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/online-payment-fraud-detection.git
    cd online-payment-fraud-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook for data exploration, preprocessing, and model training:
    ```bash
    jupyter notebook online_payment_fraud_detection.ipynb
    ```

## File Structure

- `online_payment_fraud_detection.ipynb`: Jupyter notebook containing the code for data exploration, preprocessing, and model building.
- `math_operations.py`: Module containing functions for addition and subtraction (example module).
- `main.py`: Main script demonstrating the use of the `math_operations` module (example script).
- `README.md`: This file.
- `requirements.txt`: List of dependencies required to run the project.

## Acknowledgements

- The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1).
- Special thanks to the contributors and maintainers of the libraries used in this project: NumPy, Pandas, Matplotlib, Seaborn, Scipy, Scikit-learn, Imbalanced-learn, and TensorFlow.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

