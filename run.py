import os
from os.path import join
import numpy as np
import sklearn.decomposition as decomp
from sklearn.mixture import GaussianMixture as GMM
from sklearn.svm import LinearSVC

from video_representation import VideoRepresentation
from transforms import *
from settings import *
from visualize import *


def main(already_computed_descriptors=False):
    try:
        os.mkdirs(video_descriptors_path)
    except:
        pass

    train_videos = []
    test_videos = []

    # COMPUTE DESCRIPTORS
    if not already_computed_descriptors:
        for directory in next(os.walk(data_dir))[1]:
            directory_path = join(data_dir, directory)
            print(f'\n________EXTRACTING DESCRIPTORS FROM {directory_path}')
            for filename in os.listdir(directory_path):
                filepath = join(directory_path, filename)
                if '.avi' in filename and os.path.isfile(filepath):
                    # task 1
                    trajectories_list = trajectories_from_video(filepath)
                    # task 2
                    # saves descriptors to disk
                    descriptors_from_trajectories(trajectories_list, filename)

    # TRAIN
    train_lines = []
    with open(join(data_dir, 'train.txt'), 'r') as train_f:
        train_lines = train_f.readlines()
    for l in train_lines:
        filepath, label = l.split()
        descriptor_path = join(video_descriptors_path,
                               f'{filepath.split("/")[1].replace(".avi", "-descriptors.txt")}')
        video_representation = VideoRepresentation(filepath, np.loadtxt(descriptor_path), label)
        train_videos.append(video_representation)

    all_train_descriptors = np.concatenate([v.descriptors for v in train_videos], axis=0)
    print(f'total number of train descriptors: {all_train_descriptors.shape[0]}')
    print(f'length of each train descriptor: {all_train_descriptors.shape[1]}')

    # init and fit the pca
    pca = decomp.PCA(pca_num_components)
    pca = pca.fit(all_train_descriptors)

    # transform descriptors of each video
    for v in train_videos:
        v.pca_descriptors = pca.transform(v.descriptors)

    # concatenate the pca-transformed descriptors, to not transform the whole data one extra time
    all_train_descriptors = np.concatenate([v.pca_descriptors for v in train_videos], axis=0)
    print(f'length each train descriptor after pca: {all_train_descriptors.shape[1]}')

    # learn GMM model
    gmm = GMM(n_components=gmm_n_components, covariance_type='diag')
    gmm.fit(all_train_descriptors)

    # compute fisher vectors for each train video
    for v in train_videos:
        v.fisher_vector = fisher_from_descriptors(v.pca_descriptors, gmm)
    print('calculated Fisher vectors')

    # initialize and fit a linear SVM
    svm = LinearSVC()
    svm.fit(X=[v.fisher_vector for v in train_videos], y=[v.label for v in train_videos])
    print('fitted SVM')

    # TEST
    test_lines = []
    with open(join(data_dir, 'test.txt'), 'r') as test_f:
        test_lines = test_f.readlines()
    for l in test_lines:
        filepath, label = l.split()
        descriptor_path = join(video_descriptors_path,
                               f'{filepath.split("/")[1].replace(".avi", "-descriptors.txt")}')
        video_representation = VideoRepresentation(filepath, np.loadtxt(descriptor_path), label)
        test_videos.append(video_representation)

    # reduce dimension of all test descriptors using pca fitted on train data
    for v in test_videos:
        v.pca_descriptors = pca.transform(v.descriptors)
    print('reduced dimensions of the test data')

    # calculate a fisher vector for each test video based on the gmm model fit on the train data
    for v in test_videos:
        v.fisher_vector = fisher_from_descriptors(v.pca_descriptors, gmm)
    print('calculated Fisher vectors on the test data')

    # predict the labels of the test videos
    accuracy = svm.score(X=[v.fisher_vector for v in test_videos], y=[v.label for v in test_videos])
    print(f'accuracy: {accuracy}')
    prediction = svm.predict(X=[v.fisher_vector for v in test_videos])
    for i, v in enumerate(test_videos):
        v.predicted_label = prediction[i]
    print('prediction by video: index, true label, predicted label, path\n')
    for i, v in enumerate(test_videos):
        print(f'{i}    gt: {v.label}    pred: {v.predicted_label}   {v.filepath}')


if __name__ == '__main__':
    # to test trajectories on a single video
    # trajectories_from_video('data/UnevenBars/v_UnevenBars_g01_c01.avi', vis_flow=False, vis_trajectories=True)

    main(already_computed_descriptors=False)
