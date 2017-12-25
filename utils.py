import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def make_submit_file(ids, classes, yhat, filepath):
    submission = pd.DataFrame(yhat, columns=classes)
    submission.insert(0, 'id', ids)
    submission.reset_index()
    submission.to_csv(filepath, index = False)
    
def standardize_output(yhat):
    def f(p):
        p = float(p)
        return max(min(p, 1-1e-15), 1e-15)
    f = np.vectorize(f)
    result_array = f(yhat)
    return result_array

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           
    classes = list(le.classes_)                    
    test_ids = test.id                             
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes