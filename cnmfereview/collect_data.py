'''
Author: Andrew Mocle, Lina Tran

Collect putative ROIs and their extracted fluorescence traces
from CNMF-E that have been manually reviewed and labeled into
a dataset ready for use in scikit-learn.
Example usage:

$ cd /path/to/cnmfe-reviewer
$ python cnmfrereview/collect_data.py file1.mat file2.mat file3.mat -hy 648 -wx 486

'''

import numpy as np
import os
from scipy.io import loadmat
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Collect Scored ROIs into Machine\
         Learning Format. Note, use the original .mat file and not the _keep.mat \
         version generated after applying your reviews')
    parser.add_argument('input', help='files you have manually reviewed \
        that you want to aggregate into a dataset', nargs='+')
    parser.add_argument('-hy', '--height', type=int, help="ROI image height in pixels")
    parser.add_argument('-wx', '--width', type=int, help="ROI image width in pixels")
    parser.set_defaults(height=648, width=486)
    return parser.parse_args()


if __name__== '__main__':
    args = get_args()

    data_Craw = []    # raw calcium trace
    data_A = []    # spatial footprint
    scores = []    # labels from manual review of ROIs from CNMFE
    height = args.height
    width = args.width

    for filename in tqdm(range(len(args.input))):
        data = loadmat(args.input[filename])
        # spatial downsample factor
        ds = data['ssub']

        data_Craw.append(data['C_raw'][:, :])

        A = data['A'].transpose(1,0)
        A = A.reshape((
            data['C_raw'].shape[0],
            int(height/ds),
            int(width/ds)
        ))

        data_A.append(A)


        for tt in range(data['C_raw'].shape[0]):
            # ROI labels (keep, 1, or exclude, 0)
            if tt in data['keep']:
                scores.append(1)
            else: 
                scores.append(0)

    
    np.save('data_Craw.npy', np.concatenate(data_Craw))
    np.save('data_A.npy', np.concatenate(data_A))
    np.save('scores.npy', scores)
