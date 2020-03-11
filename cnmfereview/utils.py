import numpy as np
import os

def crop_image(img, peak, margin):
    new_img = np.zeros((margin*2, margin*2))
    img = img[peak[0]-margin:peak[0]+margin, peak[1]-margin:peak[1]+margin]
    new_img[:img.shape[0], :img.shape[1]] = img
    return new_img
    
def crop_footprint(spatial, final_size):
    args = spatial.reshape(spatial.shape[0],-1).argmax(-1)
    maxima = np.unravel_index(args, spatial.shape[-2:])
    peak_coor = list(zip(maxima[0], maxima[1]))
    
    cropped_spatial = np.zeros((spatial.shape[0], final_size*2, final_size*2))
    for ix, i in enumerate(peak_coor):
        cropped_spatial[ix] = crop_image(spatial[ix].squeeze(), i, final_size)
    
    return cropped_spatial

def roundup(x):
     return x if x % 100 == 0 else x +100 - x % 100

def normalize_traces(traces, pad, longest):
    # normalize trace values by the max value for that recording
    traces_norm = (traces/[max(i) if max(i)!=0 else 1 for i in traces ])
    if longest is not None:
        longest_trace = longest
        traces_norm = np.array([x[:longest] for x in traces_norm])
    else: 
        longest_trace = roundup(max([i.shape[0] for i in traces])) 


    # pad arrays with 0 to make all the same shape
    if pad:
        traces_norm = np.array([np.pad(x,
                                    (0, longest_trace - x.shape[0]),
                                     'constant') for x in traces_norm])

    return traces_norm


def load_data(directory, dtype=np.float32):
    """
    Load the spatial footprints and trace data from directory.
    If processed files don't exist, preprocess here and load.
    """
    # check if processes files exist, if not, load and preprocess the raw data
    preprocess = not np.all([os.path.exists(f"{directory}/data_A_cropped.npy"),
                  os.path.exists(f"{directory}/data_traces_comb.npy")])
    
    if preprocess:
        # spatial footprints + crop
        spatial = np.load(f"{directory}/data_A.npy").astype(dtype)
        spatial = crop_footprint(spatial, final_size=40)
        
        # traces
        trace_raw = np.load(f"{directory}/data_Craw.npy", 
                            allow_pickle=True).astype(dtype)
        trace_d = np.load(f"{directory}/data_C.npy",
                          allow_pickle=True).astype(dtype)
        
        # normalize trace data
        traces = [normalize_traces(trace, pad=True, longest=500) 
                  for trace in [trace_raw, trace_d]]
        # stack the traces
        traces = np.array([np.vstack([i,traces[1][ix]]).T 
                             for ix,i in enumerate(traces[0])])
        
        # save processed files
        np.save(f'{directory}/data_A_cropped.npy', spatial)
        np.save(f'{directory}/data_traces_comb.npy',traces)
                           
    else:
        # load spatial and trace data
        spatial = np.load(f'{directory}/data_A_cropped.npy').astype(dtype)
        traces = np.load(f'{directory}/data_traces_comb.npy').astype(dtype)
    
    # load the labels/target data
    targets = np.load(f'{directory}/scores.npy').astype(dtype)
    
    return spatial, traces, targets


def set_up_remote_job(directory, feature='combined', split=0.2):
    # set random seed (sklearn)
    seed = 23

    # prepare data
    spatial, traces_n, targets = load_data(directory)
    # array shapes need to work with sklearn 
    px = spatial.shape[1]
    spatial = np.reshape(spatial, (len(spatial), px*px))
    traces_n = traces_n[:,:,0] #raw
    dt = {'spatial': spatial,
            'trace': traces_n,
            'combined': np.concatenate((spatial, traces_n), axis=1)}

    data = dt[feature]

    # train/test split
    from sklearn.model_selection import StratifiedShuffleSplit 

    sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=seed)
    
    for train_index, test_index in sss.split(data, targets):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
    
    print("Training and test data loaded")

    return x_train, x_test, y_train, y_test
    
