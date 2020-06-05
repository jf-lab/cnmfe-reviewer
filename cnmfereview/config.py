# experiment identifier
# up to you how you want to format, no spaces or special chars
exp_id = "20200202-exp-name-example"

# the dimensions of your field of view image in pixels
img_shape = { 
    "x": 486,  # x-dim in pixels of your spatial footprints
    "y": 648  # y-dim in pixels of your spatial footprints
}

# the xy dimensions in pixels of your cropped images
# should be about 2x larger than the largest ROI
img_crop_size = {"x": 40,
                 "y": 40}


# the number of frames to extract from each trace
max_trace_len = 500  

