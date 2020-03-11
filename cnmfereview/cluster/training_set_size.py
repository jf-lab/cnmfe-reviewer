import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cnmfereview as cr
import pandas as pd
from sklearn.model_selection import train_test_split

def job_script(data):
    repeats = 10
    groups = 2 # spatial and traces
    indices = range(repeats*groups)
    accuracies = pd.DataFrame(index=indices,
                              columns=['Group',
                                       'Repeat',
                                       'Accuracy'])

   row = 0
   for i in range(repeats):
       seed = np.random.randint(1,1000)
       results = np.zeros(2)
       for group in data:
           subset = data[group]
           x_train, x_test, y_train, y_test = train_test_split(subset[0][0],
                                                               subset[0][1],
                                                               test_size=test_size,
                                                               random_state=seed)
           clf = svm.SVC()
           clf.fit(


def main(directory):
    # {"Spatial": [[x_train, y_train], [x_test, y_test]],
    #  "Trace": [[x_train, y_train], [x_test, y_test]]}      
    data = set_up_remote_job(directory)
    job_script()


if __name__ == '__main__':
    import sys
    tmp_directory = sys.argv[-1]
    main(tmp_directory)
        
