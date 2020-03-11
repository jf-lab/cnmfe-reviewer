import numpy as np
import matplotlib.pyplot as plt
import cnmfereview as cr
import pandas as pd
from sklearn.model_selection import train_test_split


def job_script(data):
   

def main(directory):
    # {"Spatial": [[x_train, y_train], [x_test, y_test]],
    #  "Trace": [[x_train, y_train], [x_test, y_test]]}      
    data = set_up_remote_job(directory)
    job_script()


if __name__ == '__main__':
    import sys
    tmp_directory = sys.argv[-1]
    main(tmp_directory)
        
