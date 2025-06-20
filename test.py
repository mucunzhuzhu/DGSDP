import glob
import os
import pickle
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from DGSDP import DGSDP
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

random.seed(123)

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    # feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    if opt.feature_type == "jukebox":
        feature_func = juke_extract
    elif opt.feature_type == "baseline":
        feature_func = baseline_extract

    sample_length = opt.out_length
    sample_length = 5
    sample_size = int(sample_length / 2.5) - 1 # 11

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            # print('dir',dir)
            # print('len(file_list)',len(file_list))
            # print('len(juke_file_list)',len(juke_file_list))
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    print("len(all_cond)",len(all_cond))

    model = DGSDP(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        styles = []
        for j in range(len(all_filenames[i])):
            wav = os.path.basename(all_filenames[i][j])
            style = wav.split('_')[4]
            style = style[:-1]
            styles.append(style)

        style_features = []
        for j in range(len(styles)):
            path = './data/style_clip/' + styles[j] + '.pkl'
            with open(path, 'rb') as f:
                data = pickle.load(f)
                style_features.append(data[styles[j]])

        style_features = torch.tensor([item.cpu().detach().numpy() for item in style_features]).cuda()
        batchsize, original_dim = style_features.size()
        style_tensor = style_features.unsqueeze(1).expand(batchsize, 150, original_dim)
        print("style_tensor.shape",style_tensor.shape)
        style_tensor = style_tensor.detach().cpu()
        # style_tensor = style_tensor.to(self.accelerator.device)
        all_cond[i] = torch.cat((all_cond[i], style_tensor), dim=-1)

        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render # render = True
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    # opt.use_cached_features = True
    # opt.feature_cache_dir = "cached_features"
    # opt.save_motions = True
    # opt.motion_save_dir = "SMPL-to-FBX/motions"
    test(opt)

    # opt = parse_test_opt()
    # motion_save_dir = opt.motion_save_dir
    # for i in range(100):
    #     opt.motion_save_dir = motion_save_dir + '_' +str(i)
    #     test(opt)


