import numpy as np
import os
from os.path import join

from trajectory import Trajectory
from settings import *
from visualize import *


def trajectories_from_video(path, vis_flow=False, vis_trajectories=False,
                            W=W,  # sampling grid spacing
                            L=L,  # maximum length of a trajectory
                            static_displacement_thresh=static_displacement_thresh,  # static if the sum of all displacements' norms is lower
                            max_single_displacement=max_single_displacement,  # max percentage a single displacement has in a trajectory
                            ):
    # read the video at 'path' frame by frame
    capture = cv.VideoCapture(path)
    r, current_frame = capture.read()
    current_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    height, width = current_frame.shape  # video resolution

    flow = None
    flow_list = []

    grid_xs = list(range(0, width, W))
    grid_ys = list(range(0, height, W))

    frame_i = 0
    trajectories = []
    complete_trajectories = []
    shape_descriptors = []

    print(f'{path}: extracting trajectories')
    while True:
        # pad the frame for extracting tubes around the borders
        gx = cv.Sobel(current_frame, cv.CV_32F, 1, 0)
        gy = cv.Sobel(current_frame, cv.CV_32F, 0, 1)
        # current_frame_padded = cv.copyMakeBorder(current_frame, top=N2, bottom=N2, left=N2, right=N2,
        #                                          borderType=cv.BORDER_REFLECT)
        r, next_frame = capture.read()
        if r:
            next_frame = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(current_frame, next_frame, flow, pyr_scale=None, levels=1,
                                               winsize=of_winsize, iterations=10, poly_n=5, poly_sigma=1.1,
                                               flags=None)  # same values
            # round the values so each pixel is 'moving' to a particular pixel in the next frame
            flow = np.round(flow)

            # pad the flow image for extracting optical flow tubes
            flow_x_padded = cv.copyMakeBorder(flow[:, :, 0], top=N2, bottom=N2, left=N2, right=N2,
                                              borderType=cv.BORDER_REFLECT)
            flow_y_padded = cv.copyMakeBorder(flow[:, :, 1], top=N2, bottom=N2, left=N2, right=N2,
                                              borderType=cv.BORDER_REFLECT)
            flow_padded = np.zeros((height + N, width + N, 2), dtype='int32')
            flow_padded[:, :, 0] = flow_x_padded
            flow_padded[:, :, 1] = flow_y_padded

            # MBHx from flow_x, padded. MBHx is the derivatives in x and y of the x (or u) component of the flow
            u_x = cv.Sobel(flow_x_padded, cv.CV_32F, 1, 0)  # x derivative of the u component of the flow
            u_y = cv.Sobel(flow_x_padded, cv.CV_32F, 0, 1)  # y derivative of the u component of the flow
            mbhx_padded = np.zeros((height + N, width + N, 2), dtype='float32')
            mbhx_padded[:, :, 0] = u_x
            mbhx_padded[:, :, 1] = u_y

            # MBHy from flow_y, padded. MBHy is the derivatives in x and y of the y (or v) component of the flow
            v_x = cv.Sobel(flow_y_padded, cv.CV_32F, 1, 0)  # x derivative of the u component of the flow
            v_y = cv.Sobel(flow_y_padded, cv.CV_32F, 0, 1)  # y derivative of the u component of the flow
            mbhy_padded = np.zeros((height + N, width + N, 2), dtype='float32')
            mbhy_padded[:, :, 0] = v_x
            mbhy_padded[:, :, 1] = v_y

            # pad and combine the grad images
            gx_padded = cv.copyMakeBorder(gx, top=N2, bottom=N2, left=N2, right=N2,
                                              borderType=cv.BORDER_REFLECT)
            gy_padded = cv.copyMakeBorder(gy, top=N2, bottom=N2, left=N2, right=N2,
                                          borderType=cv.BORDER_REFLECT)
            g_padded = np.zeros((height + N, width + N, 2), dtype='float32')
            g_padded[:, :, 0] = gx_padded
            g_padded[:, :, 1] = gy_padded

            # get coordinates of all the corners that pass the threshold, not constrained by min distance between
            corners = cv.goodFeaturesToTrack(current_frame, maxCorners=0,  # unlimited number of points to be kept
                                             qualityLevel=corner_quality_level,
                                             minDistance=1  # between the corners, every px can be one
                                             )
            # make corner coordinates discrete so we can use it as indices
            corners = np.int0(corners)

            if vis_flow:
                # VISUALIZE THE FLOW
                cv.imshow('flow', draw_flow(current_frame, flow, step=W))
                cv.waitKey(1)

            # init new trajectories at corners, if 1) they are on the grid, 2) no tracked points in WxW
            for c in corners:
                x, y = c.ravel()
                if x in grid_xs and y in grid_ys:  # if the corner is on the grid
                    if not any(np.abs(y - t.coords[-1][1]) <= W and
                               np.abs(x - t.coords[-1][0]) <= W
                               for t in trajectories):  # if no tracked points in WxW
                        trajectories.append(Trajectory(y=y, x=x, start_frame_num=frame_i))

            # continue existing trajectories
            for t in trajectories:
                # the last point of the trajectory
                tracked_point_y = int(t.coords[-1][1])
                tracked_point_x = int(t.coords[-1][0])
                # if the tracked point went out of bounds - remove the trajectory
                if tracked_point_x not in range(width + 1) or tracked_point_y not in range(height + 1):
                    trajectories.remove(t)
                    continue
                try:
                    # u and v components of the flow at this point
                    delta_y = flow[tracked_point_y, tracked_point_x, 1]
                    delta_x = flow[tracked_point_y, tracked_point_x, 0]
                except IndexError:
                    # remove the trajectory if it went out of the boundaries of the image
                    trajectories.remove(t)
                else:
                    # fill in the tube slices for mbhx, mbhy, flow and gradient images
                    # y: y + N and not y - N2: y + N2 because the image was padded
                    g_volume_slice = g_padded[tracked_point_y: tracked_point_y + N,
                                              tracked_point_x: tracked_point_x + N]
                    t.g_volume.append(g_volume_slice)

                    of_volume_slice = flow_padded[tracked_point_y: tracked_point_y + N,
                                                  tracked_point_x: tracked_point_x + N,
                                                  :]
                    t.of_volume.append(of_volume_slice)

                    mbhx_volume_slice = mbhx_padded[tracked_point_y: tracked_point_y + N,
                                                    tracked_point_x: tracked_point_x + N,
                                                    :]
                    t.mbhx_volume.append(mbhx_volume_slice)
                    mbhy_volume_slice = mbhy_padded[tracked_point_y: tracked_point_y + N,
                                                    tracked_point_x: tracked_point_x + N,
                                                    :]
                    t.mbhy_volume.append(mbhy_volume_slice)

                    # continue the trajectory to the next frame if it's length is less than L
                    t.add(delta_y=delta_y, delta_x=delta_x)
                    if len(t.coords) > L:
                        # take out the trajectory to the 'complete_trajectories' list if its length reached L
                        t.coords = t.coords[:-1]  # remove last (L+1'th) coordinate pair referring to the next frame
                        complete_trajectories.append(t)
                        trajectories.remove(t)

            if vis_trajectories:
                # VISUALIZE THE TRAJECTORIES
                cv.imshow('flow', draw_trajectories(current_frame, trajectories))
                cv.waitKey(1)

            flow_list.append(flow)  # append the flow to a list of flows to extract descriptors later
            current_frame = next_frame
            frame_i += 1

        else:
            break

    print(f'{len(complete_trajectories)} complete trajectories before post-processing')

    # POST-PROCESSING THE COMPLETE TRAJECTORIES
    final_trajectories = []
    for t in complete_trajectories:
        # removing trajectories with sudden large displacements
        # 'if the displacement vector between 2 frames is > 70% of the overall displacement of the trajectory'
        # understanding the 'overall displacement' as the sum of displacements' norms, not as "end minus start"
        # to preserve looping trajectories
        overall_displacement = sum([np.linalg.norm(d) for d in t.displacement_vectors])
        if any(np.linalg.norm(d) > max_single_displacement * overall_displacement for d in t.displacement_vectors):
            # complete_trajectories.remove(t)
            continue

        # removing 'static' trajectories - not specified in the paper
        # can be ones with the sum of the all displacements' norms is lower than static_displacement_thresh
        # set it to 1 to remove just the trajectories that do not move at all
        if overall_displacement < static_displacement_thresh:
            # complete_trajectories.remove(t)
            continue

        # converting tubes to numpy arrays
        t.g_volume = np.concatenate([f.reshape(1, N, N, 2) for f in t.g_volume], axis=0)
        t.of_volume = np.concatenate([f.reshape(1, N, N, 2) for f in t.of_volume], axis=0)
        t.mbhx_volume = np.concatenate([f.reshape(1, N, N, 2) for f in t.mbhx_volume], axis=0)
        t.mbhy_volume = np.concatenate([f.reshape(1, N, N, 2) for f in t.mbhy_volume], axis=0)

        # remove trajectories for which either of the volumes turned out empty
        if isinstance(t.g_volume, list) or isinstance(t.of_volume, list) \
                or isinstance(t.mbhx_volume, list) or isinstance(t.mbhy_volume, list):
            # complete_trajectories.remove(t)
            continue

        # calculating shape descriptors on the good remaining trajectories
        t.trajectory_shape = np.array(t.displacement_vectors).flatten()\
                             / sum([np.linalg.norm(d) for d in t.displacement_vectors])

        final_trajectories.append(t)

    print(f'{len(final_trajectories)} complete trajectories after post-processing')

    return final_trajectories


# TASK 2
def descriptor_from_volume(volume, small_vectors_bin=False):
    time_tubes = np.split(volume, n_tau, axis=0)
    y_tubes = [item for tube_list in
               [np.split(time_tube, n_sigma, axis=1) for time_tube in time_tubes]
               for item in tube_list]
    del time_tubes
    tubes = [item for tube_list in
             [np.split(y_tube, n_sigma, axis=2) for y_tube in y_tubes]
             for item in tube_list]
    del y_tubes

    if small_vectors_bin:
        effective_bins = bins + 1
    else:
        effective_bins = bins

    # initialize the descriptor in a shape to allow one tube's descriptor on top of the other one
    d = np.zeros((len(tubes), effective_bins), dtype='float32')

    for tube_i, tube in enumerate(tubes):
        tube_d = np.zeros(effective_bins, dtype='float32')

        for frame_i in range(tube.shape[0]):
            frame_d = np.zeros_like(tube_d)
            mag, ang = cv.cartToPolar(
                tube[frame_i, :, :, 0].astype(dtype='float32'), tube[frame_i, :, :, 1].astype(dtype='float32')
            )
            ang = np.degrees(ang)  # convert to degrees

            if small_vectors_bin:
                # populate main bins
                for bin_i in range(bins):
                    this_bin_magnitudes = np.where(
                        (ang >= bin_angles[bin_i]) & (ang < bin_angles[bin_i + 1]) & (ang > small_flow_magnitude),
                        ang, 0
                    )
                    frame_d[bin_i] = np.sum(this_bin_magnitudes)

                # last bin with small magnitudes
                frame_d[-1] = np.sum(
                    np.where(ang <= small_flow_magnitude, True, False)
                )

            else:
                # populate bins
                for bin_i in range(bins):
                    this_bin_magnitudes = np.where(
                        (ang >= bin_angles[bin_i]) & (ang < bin_angles[bin_i + 1]),
                        ang, 0
                    )
                    frame_d[bin_i] = np.sum(this_bin_magnitudes)

            # sum up the descriptors of different frames of the tube
            tube_d = np.add(tube_d, frame_d)

        d[tube_i, :] = tube_d  # stack the descriptors of individual tubes

    d = d.flatten()  # concatenate the HoFs of tubes
    d = d / (np.linalg.norm(d) + 0.1)  # normalize the vector
    return d


def descriptors_from_trajectories(trajectories, filename):
    video_descriptors = np.zeros((len(trajectories), len_descriptor), dtype=np.float32)
    for t_i, t in enumerate(trajectories):
        try:
            t.hog = descriptor_from_volume(t.g_volume)
        except:
            print('a')
        t.hof = descriptor_from_volume(t.of_volume, small_vectors_bin=True)
        t.mbhx = descriptor_from_volume(t.mbhx_volume)
        t.mbhy = descriptor_from_volume(t.mbhy_volume)
        # concatenate all descriptors
        video_descriptors[t_i] = np.concatenate((t.trajectory_shape, t.hog, t.hof, t.mbhx, t.mbhy))

    descriptors_txt_path = os.path.join(video_descriptors_path, f'{filename.split(".")[0]}-descriptors.txt')

    # save the descriptors to disk
    np.savetxt(descriptors_txt_path, video_descriptors)
    print(f'descriptors written to {descriptors_txt_path}\n________')

    return video_descriptors


# TASK 3
def fisher_from_descriptors(vectors, gmm):
    T = vectors.shape[0]
    D = vectors.shape[1]

    # posterior probabilities
    g = gmm.predict_proba(vectors)

    D_h = (np.sum(g, axis=0) - gmm.weights_) / np.sqrt(gmm.weights_)
    D_h = D_h / T  # normalize by number of trajectories

    D_mu_list = []
    for k in range(gmm_n_components):
        D_mu_k = np.zeros(D, dtype=np.float32)
        for t in range(T):
            D_mu_k_t = g[t, k] * (vectors[t, :] - gmm.means_[k]) / gmm.covariances_[k]
            D_mu_k = np.add(D_mu_k, D_mu_k_t)

        D_mu_k = D_mu_k / np.sqrt(gmm.weights_[k])
        D_mu_list.append(D_mu_k)
    D_mu = np.array(D_mu_list).flatten() / T  # K x D flattened, normalized by number of trajectories

    D_sigma_list = []
    for k in range(gmm_n_components):
        D_sigma_k = np.zeros(D, dtype=np.float32)
        for t in range(T):
            D_sigma_k_t = np.sqrt(2) * g[t, k] * (
                    (vectors[t, :] - gmm.means_[k]) ** 2 / gmm.covariances_[k] ** 2 - 1
            )
            D_sigma_k = np.add(D_sigma_k, D_sigma_k_t)

        D_sigma_k = D_sigma_k / np.sqrt(gmm.weights_[k])
        D_sigma_list.append(D_sigma_k)
    D_sigma = np.array(D_sigma_list).flatten() / T  # K x D flattened, normalized by number of trajectories

    fisher_vector = np.concatenate([D_h, D_mu, D_sigma])  # (2 * D + 1) * K
    fisher_vector = np.sqrt(np.abs(fisher_vector)) * np.sign(fisher_vector)  # power normalize the entire vector

    return fisher_vector
