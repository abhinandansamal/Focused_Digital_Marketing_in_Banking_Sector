# Focused_Digital_Marketing_in_Banking_Sector

### Objective
Build a machine learning model to perform focused digital marketing by predicting the potential customers who will convert from liability customers to asset customers.

### Data Description
The dataset has 2 CSV files:
    * Data1 - 5000 rows and 8 columns
    * Data2 - 5000 rows and 7 columns

**Attributes:**

1. **ID**: Customer ID
2. **Age**: Customer's approximate age
3. **CustomerSince**: Customer of the bank since. [unit is masked]
4. **HighestSpend**: Customer's highest spend so far in one transaction. [unit is masked]
5. **ZipCode**: Customer's zip code
6. **HiddenScore**: A score associated to the customer which is masked by the bank as an IP
7. **MonthlyAverageSpend**: Customer's monthly average spend so far. [unit is masked]
8. **Level**: A level associated to the customer which is masked by the bank as an IP.
9. **Mortgage**: Customer's mortgage. [unit is masked]
10. **Security**: Customer's security asset with the bank. [unit is masked]
11. **FixedDepositAccount**: Customer's fixed deposit account with the bank. [unit is masked]
12. **InternetBanking**: If the customer uses internet banking.
13. **CreditCard**: If the customer uses bank's credit card.
14. **LoanOnCard**: If the customer has a loan on credit card.

### Tech stack
- **Language** - Python
- **Libraries** - NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, imblearn, pickle

### Approach
1. Importing the required libraries and reading the dataset.
    * Merging of the two datasets
    * Understanding the dataset

2. Exploratory Data Analysis (EDA)

3. Feature Engineering
    * Dropping of unwanted columns
    * Removal of null values
    * Checking for multi-collinearity and removal of highly correlated features

4. Model Building
    * Performing train test split
    * Logistic Regression Model
    * Weighted Logistic Regression Model
    * Naive Bayes Model
    * Support Vector Machine Model
    * Decision Tree Classifier
    * Random Forest Classifier

5. Model Validation
    * Accuracy score
    * Confusion matrix
    * Area Under Curve (AUC)
    * Recall score
    * Precision score
    * F1-score

6. Handling unbalanced data using imblearn

7. Hyperparameter Tuning (GridSearchCV)
    * For Support Vector Machine Model

8. Creating the final model and making predictions

9. Save the model with the highest accuracy in the form of a pickle file

### Project Structure


