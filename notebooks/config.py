# experiment identifier
# up to you how you want to format, no spaces or special chars
exp_id = "cr_tutorial"

data_paths = {
    # modify these as necessary for your data
    "data_directory": "../data",
    "spatial": "data_A.npy",
    "trace": "data_Craw.npy",
    "targets": "scores.npy"
}

# the dimensions of your field of view image in pixels
img_shape = { 
    "x": 486,  # x-dim in pixels of your spatial footprints
    "y": 648  # y-dim in pixels of your spatial footprints
}

# the xy dimensions in pixels of your cropped images
# should be about 2x larger than the largest ROI
img_crop_size = {"x": 80,
                 "y": 80}


# the number of frames to extract from each trace
max_trace_len = 500  

