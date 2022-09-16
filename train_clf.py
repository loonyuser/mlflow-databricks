import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import train_test_split


import mlflow
import mlflow.sklearn

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    loan_approval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cust_data_processed.csv')
    cust_data = pd.read_csv( loan_approval_path )
    X = cust_data.drop(labels = ['loan_approval_status'], axis = 1)

    y = cust_data['loan_approval_status']
    # Split the data into training and test sets. (0.7, 0.3) split.  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 123)
      
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    min_samples_split = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    min_samples_leaf  = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    with mlflow.start_run():          
        model = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split, 
                                       min_samples_leaf = min_samples_leaf)
        model.fit(X_train, y_train)
        predictions =  model.predict(X_test)
        predictions_proba = model.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, predictions)
        test_precision_score = precision_score(y_test, predictions)
        test_recall_score = recall_score(y_test, predictions)
        test_f1_score = f1_score(y_test, predictions)
        auc_score = roc_auc_score(y_test, predictions_proba[:,1])
        metrics = {'Test_accuracy': test_accuracy, 'Test_precision_score': test_precision_score,
                   'Test_recall_score': test_recall_score,'Test_f1_score': test_f1_score, 'AUC_score': auc_score}
    
          
        mlflow.log_metrics(metrics)
    
        mlflow.set_tag('Classifier', 'RF-tuned parameters')
       
        mlflow.sklearn.log_model(model, 'RF-tuned parameters')
        
        

        
