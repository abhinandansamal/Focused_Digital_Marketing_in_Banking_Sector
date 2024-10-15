# Focused_Digital_Marketing_in_Banking_Sector

## Objective
Build a machine learning model to perform focused digital marketing by predicting the potential customers who will convert from liability customers to asset customers.

## Tech stack
- **Language** - Python
- **Libraries** - NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, imblearn, pickle

## Approach
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

## Project Structure
```
Focused_Digital_Marketing_In_Banking_Sector/
│
├── data/                          # Folder containing datasets
│   ├── Data1.csv
│   └── Data2.csv
│
├── images/                        # Folder containing images for documentation or reports
│   └── tree.png
│
├── notebooks/                     # Folder for Jupyter Notebooks
│   └── focused_digital_marketing_in_banking.ipynb
│
├── output/                        # Folder for storing model outputs
│   └── finalized_model.sav        # The finalized model saved for future predictions
│
├── src/                           # Source code folder
│   ├── app.py                     # Main application file
│   ├── ml_pipeline/               # ML pipeline components
│       ├── grid_model.py          # Script for performing GridSearchCV on multiple models
│       ├── model_evaluation.py    # Script for evaluating models (accuracy, recall, etc.)
│       ├── train_model.py         # Script for training models
│       └── utils.py               # Utility functions for the project
│
├── .gitignore                     # Git ignore file
├── LICENSE                        # License information
├── README.md                      # Project description and instructions (this file)
├── requirements.txt               # Python package dependencies
```

## Installation
1. Clone the repository:

    ``git clone https://github.com/abhinandansamal/Focused_Digital_Marketing_in_Banking_Sector.git``

2. Navigate into the project directory:

    ``cd focused_digital_marketing_in_banking``

3. Create Virtual Environment

    ``conda create env --name <env_name>``

4. Install the required dependencies: Use the requirements.txt file to install all the necessary packages.

    ``pip install -r requirements.txt``


## Data
The `data/` folder contains the CSV files required for training and evaluating machine learning models. The provided datasets (`Data1.csv`, `Data2.csv`) contain the features relevant to digital marketing campaigns in the banking domain.

## Model Training and Evaluation
* The project supports multiple machine learning models for training, including `Logistic Regression`, `Naive Bayes`, `Support Vector Machine`, `Decision Tree`, and `Random Forest`.

* Data imbalance is handled by using `SMOTE` & `RandomUnderSampler` techniques. 

* Grid Search Cross-Validation (`GridSearchCV`) is used for hyperparameter tuning to select the best model based on accuracy & recall metrics.

## Key Components
- `grid_model.py`: This script implements GridSearchCV for hyperparameter tuning on different models.
- `train_model.py`: This script trains multiple models and selects the best one based on performance metrics.
- `model_evaluation.py`: This script provides functions for evaluating models using different metrics such as accuracy.
- `utils.py`: Contains utility functions used throughout the pipeline.

## Usage
1. **Running the Jupyter Notebook**: You can explore and interact with the project through the Jupyter notebook:
    ```jupyter notebook notebooks/focused_digital_marketing_in_banking.ipynb```

2. **Training and Evaluating Models**: You can train models and evaluate them using the scripts provided in the src/ml_pipeline/ directory. For example, to run a grid search on model:
    ```python src/ml_pipeline/grid_model.py```

3. **Saving and Loading Models**: The trained model can be saved to the output/ directory, where the file finalized_model.sav is an example of a trained and saved model.

## Conclusion

* Models are built using Logistic Regression, Naive Bayes, Support Vector Machine Classifier, Decision Tree Classifier and Random Forest Classifier. The data set is highly imbalance hence accuracy can't a good measure, Hence I have used precision, Recall, and AUC for determining better model.

* I have used class weight technique to handle data imbalance and observed that the model performance improved by considering class weight.

* Scaling/data transformation plays a major role when working on SVM.

* I have used undersampling and oversampling techniques like SMOTE to handle data imbalance.

* Hyper parameter tuning is done using GridSearchCV.

* Finally RandomForestClassifier became the best model, based on the performance metrics.