from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB # using Gaussian algorithm from Naive Bayes
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter(action='ignore')


# define a function to perform grid search cv
def grid_model(X_train,y_train,param_grid,model_name):
  """
  Performs hyperparameter tuning using GridSearchCV for a given model and training data.
  
  Parameters:
  -----------
  X_train : pandas.DataFrame or numpy.ndarray
      Training feature set.
  y_train : pandas.Series or numpy.ndarray
      Target labels for the training set.
  param_grid : dict
      Dictionary specifying the hyperparameters to be tested during grid search. 
      The keys should match the parameter names of the selected model, and the values should be lists of parameter settings to try.
  model_name : str
      The name of the model to use for grid search. Options include:
      - 'logistic_reg': Logistic Regression
      - 'naive_bayes': Naive Bayes (Gaussian)
      - 'svm_model': Support Vector Machine (SVC)
      - 'decision_tree': Decision Tree Classifier
      - 'rfcl': Random Forest Classifier
  Returns:
  --------
  final_predictor : sklearn model object
      The model with the best hyperparameters found by GridSearchCV.
  """
  model_dict = {
    'logistic_reg' : LogisticRegression,
    'naive_bayes'  : GaussianNB,
    'svm_model'    : SVC,
    'decision_tree': DecisionTreeClassifier,
    'rfcl'         : RandomForestClassifier,
     }

  model = model_dict[model_name]()
  model.fit(X_train, y_train)

  grid_search = GridSearchCV(model, param_grid, refit=True, verbose=42)
  grid_search.fit(X_train,y_train)
  final_predictor = grid_search.best_estimator_
  
  return final_predictor