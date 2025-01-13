# Credit Card Fraud Detection

This project predicts fraudulent credit card transactions using machine learning models. It is designed to assist banks in identifying potential fraud and safeguarding customer transactions. The project handles highly imbalanced data effectively using techniques like undersampling and applies models like Logistic Regression and XGBoost for accurate predictions.

## Features
- Data preprocessing including feature scaling and transformation
- Class imbalance handling using undersampling
- Logistic Regression and XGBoost with hyperparameter tuning
- Evaluation metrics: ROC-AUC, F1-Score, Sensitivity, and Specificity
- Streamlit-based interactive web app for fraud prediction

## Files
- `notebook.ipynb`: Python notebook containing the complete model pipeline
- `streamlit_app.py`: Streamlit app for fraud prediction
- `best_model.pkl`: Saved model for prediction
- `scaler.pkl` and `transformer.pkl`: Preprocessing objects
- `creditcard.csv`: Dataset containing credit card transactions

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/xeno1919/Credit-Card-Fraud-Detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
5. Open the app in your browser at `http://localhost:8501`.

## Dataset
The dataset contains transactions made by European cardholders over two days in September 2013. It includes 284,807 transactions, of which only 0.172% are fraudulent. The features are PCA-transformed.

## Key Metrics
- **Train AUC**: 0.99 (Logistic Regression), 1.00 (XGBoost)
- **Test AUC**: 0.97 (Logistic Regression), 0.98 (XGBoost)
- Effective handling of imbalanced data with recall-focused metrics.

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Streamlit
- Pandas, NumPy, Matplotlib, Seaborn

## Contributions
Contributions are welcome! Feel free to open an issue or create a pull request.

## License
This project is licensed under the MIT License.
