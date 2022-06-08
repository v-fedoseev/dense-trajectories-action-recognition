import os
from os.path import join

# TRACKING
W = 5  # stride of the grid for sampling new tracks
L = 15  # number of frames for each complete trajectory
of_winsize = 20  # optical flow window size

# Sh-Tomasi threshold on cornerness response function values
# 0.001 as in the paper detects too many corners with some corners in homogeneous regions
# maybe because here we use only 1 spatial scale and no smoothing, so we need a stricter threshold
corner_quality_level = 0.01
static_displacement_thresh = 5
max_single_displacement = 0.7


# TUBES AND DESCRIPTORS
N = 32  # side of the square around the tracked point (1-frame section of the tube)
N2 = int(N / 2)
n_sigma = 2  # number of tube grid cells by x and y
n_tau = 3  # number of tube grid cells in time
bins = 8  # number of bins for all the descriptors except HoF, which would be bins + 1
len_descriptor = L * 2 + 3 * bins * n_sigma * n_sigma * n_tau + (bins + 1) * n_sigma * n_sigma * n_tau
bin_angles = [int(i * 360 / bins) for i in range(bins + 1)]
small_flow_magnitude = 1


# PCA AND GMM
pca_num_components = 64
gmm_n_components = 3


# PATHS
video_descriptors_path = join('out', 'video_descriptors')
data_dir = 'data'
