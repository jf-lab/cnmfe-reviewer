import numpy as np
import os
import config as cfg

def crop_image(img, peak, x_margin, y_margin):
    """
    Crops an image centered on the peak coordinates, with a 
    margin of x_margin and y_margin on both sides, in both 
    dimensions. 
    
    The resultant image will be of shape:
    [x_margin*2, y_margin*2]

    Args:
        img ([type]): [de   scription]
        peak ([type]): [description]
        x_margin ([type]): [description]
        y_margin ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_img = np.zeros((x_margin*2, y_margin*2))
    img = img[peak[0]-x_margin:peak[0]+x_margin, peak[1]-y_margin:peak[1]+y_margin]
    new_img[:img.shape[0], :img.shape[1]] = img

    return new_img
    

def crop_footprint(
        spatial,
        crop_dim=[cfg.img_crop_size['x'],
                  cfg.img_crop_size['y']]):
    """
    Crop each spatial footprint based on peak of each ROI.

    Args:
        spatial (np.ndarray): Spatial footprints from whole
            field of view.
        crop_dim ([int, int]): [X, Y] dimensions, in pixels
            of the final cropped image. If only one value provided,
            the final cropped image will be a square of shape
            crop_dim x crop_dim
    Returns:
        cropped_spatail (np.ndarray): Array of cropped spatial
            footprints of shape (spatial[0], crop_dim[0], crop_dim[1]) 
    """
    if len(crop_dim) == 1:
        x = crop_dim
        y = crop_dim
    elif len(crop_dim) == 2:
        x = crop_dim[0]
        y = crop_dim[1]

    args = spatial.reshape(spatial.shape[0],-1).argmax(-1)
    maxima = np.unravel_index(args, spatial.shape[-2:])
    peak_coor = list(zip(maxima[0], maxima[1]))
    
    cropped_spatial = np.zeros((spatial.shape[0],
                                x,
                                y))
    for ix, peak in enumerate(peak_coor):
        cropped_spatial[ix] = crop_image(spatial[ix].squeeze(),
                                         peak,
                                         x/2,
                                         y/2)
    
    return cropped_spatial


def normalize_traces(traces, pad, longest=None):
    # convenience function to round up trace array lengths 
    # to the nearest 100th
    def roundup(x):
        return x if x % 100 == 0 else x + (100 - x % 100)
        
    # normalize trace values by the max value for that recording
    traces_norm = (traces/[max(i) if max(i)!=0 else 1 for i in traces ])


    if longest is not None:
        longest_trace = longest
        traces_norm = np.array([x[:longest] for x in traces_norm])
    else: 
        longest_trace = roundup(max([i.shape[0] for i in traces])) 


    # if pad=True, pad arrays with 0 values to make all the same shape
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
        spatial = crop_footprint(spatial, crop_dim=[80,80])
        
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


def setup_remote_job(directory, feature='combined', split=0.2):
    # set random seed (sklearn)
    seed = 0

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
    

def preprocess_2p_dataset(spatial, traces, targets, methods=None, 
                             feature='combined', split=0.2):  
    seed = 0
    px = spatial.shape[1]
    spatial = np.reshape(spatial, (len(spatial), px*px))
    dt = {'spatial': spatial,
            'trace': traces,
            'combined': np.concatenate((spatial, traces), axis=1)}

    data = dt[feature]

    # train/test split
    from sklearn.model_selection import StratifiedShuffleSplit 

    sss = StratifiedShuffleSplit(n_splits=1, 
                                 test_size=split, 
                                random_state=seed)
    
    for train_index, test_index in sss.split(data, targets):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        print("Training and test data loaded")
        if methods is not None:
            m_train, m_test = methods[train_index], methods[test_index]
            print("Target descriptions loaded \
                   (ground truth dataset only)")
            return x_train, x_test, y_train, y_test, m_train, m_test

        return x_train, x_test, y_train, y_test


def classification(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Create four arrays of the indices of false positives, 
    false negatives, true positives and true negatives 
    from predicted (y_pred) and target (y_true) classes.

    Args:
        y_true (np.ndarray): Array of the true target values
        y_pred (np.ndarray): Array of predicted values
    """
    def false_positive(y_true, y_pred):
        idx = np.argwhere(y_pred > y_true)
        return idx

    def false_negative(y_true, y_pred):
        idx = np.argwhere(y_true > y_pred)
        return idx
    
    def true_positive(y_true, y_pred):
        idx = np.argwhere((y_pred == y_true) & (y_true == 1))
        return idx

    def true_negative(y_true, y_pred):
        idx = np.argwhere((y_pred == y_true) & (y_true == 0))
        return idx
    
    fp = false_positive(y_true, y_pred).ravel()
    fn = false_negative(y_true, y_pred).ravel()
    tp = true_positive(y_true, y_pred).ravel()
    tn = true_negative(y_true, y_pred).ravel()
    
    return fp, fn, tp, tn


def retrieve_sp_tr(data,
                   x=cfg.img_crop_size['x'],
                   y=cfg.img_crop_size['y']):
    """
    Retreives the original spatial footprint and trace data
    from flattened and concatenated array used for classification

    Args:
        data (np.ndarray): NxM array where N is the number of cells,
            and M is the length of the flattened, concatenated
            spatial and trace data.
        x (int, optional): Original cropped image x dimension. 
            Defaults to cfg.img_shape['x]
        y (int, optional): Original cropped image y dimension. 
            Defaults to cfg.img_shape['y']

    Returns:
        spatial: original spatial images
        tr: original trace data
    """
    split = x*y # index of the end of ROI, start of trace

    spatial = data[:, :split]
    spatial = spatial.reshape((len(spatial), x, y))
    
    tr = data[:, split:]

    return spatial, tr

