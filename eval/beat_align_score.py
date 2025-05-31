import numpy as np
import pickle
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
import json
# kinetic, manual
import os
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import glob
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

music_root = './test_wavs_sliced/baseline_feats'

def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        # print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]

        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]

        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')

        # print(len(beats))
        return beat_axis


def get_mb2(key, length=None):
    path = os.path.join(music_root, key)
    if length is not None:
        beats = np.load(path)[:, 34][:][:length]
    else:
        beats = np.load(path)[:, 34]
    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]

    return beat_axis


def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb) ** 2) / 2 / 9)
    return (ba / len(music_beats))


def calc_ba_score(root):
    ba_scores = []

    it = glob.glob(os.path.join(root, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        print('pkl',pkl)
        info = pickle.load(open(pkl, "rb"))
        joint3d = info["full_pose"]

        joint3d = joint3d.reshape(-1,72)

    # for pkl in os.listdir(root):
    #     # print(pkl)
    #     if os.path.isdir(os.path.join(root, pkl)):
    #         continue
    #     joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:,
    #               :]  # shape:(length+1,72)

        dance_beats, length = calc_db(joint3d, pkl)
        print('length',length)
        pkl=  os.path.basename(pkl)
        pkl_split = pkl.split('.')[0].split('_')[1] + '_' + pkl.split('.')[0].split('_')[2] + '_' + \
                     pkl.split('.')[0].split('_')[3] + '_' + pkl.split('.')[0].split('_')[4] + '_' + \
                     pkl.split('.')[0].split('_')[5] + '_' + pkl.split('.')[0].split('_')[6]
        pkl_split2 = pkl.split('.')[0].split('_')[1]+'_'+pkl.split('.')[0].split('_')[2]+ '_'+\
                     pkl.split('.')[0].split('_')[3]+'_'+pkl.split('.')[0].split('_')[4]+'_'+\
                     pkl.split('.')[0].split('_')[5]+'_'+pkl.split('.')[0].split('_')[6]+'_' + \
                     pkl.split('.')[0].split('_')[7]
        # music_beats = get_mb(pkl_split + '.json', length)
        music_beats = get_mb2(pkl_split +'/'+pkl_split2 +'.npy', length)
        ba_scores.append(BA(music_beats, dance_beats))

    return np.mean(ba_scores)


if __name__ == '__main__':
    pred_root = 'eval/motions'
    print(calc_ba_score(pred_root))
