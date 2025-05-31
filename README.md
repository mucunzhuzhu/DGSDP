## DGSDP
**Flexible Music-Conditioned Dance Generation with Style Description Prompts**<br>
*Abstract: Dance plays an important role as an artistic form and expression in human culture, yet the creation of dance remains a challenging task. Most dance generation methods primarily rely solely on music, seldom taking into consideration intrinsic attributes such as music style or genre.  In this work, we introduce Flexible Dance Generation with Style Description Prompts (DGSDP), a diffusion-based framework for suitable for diversified tasks of dance generation by fully leveraging the semantics of music style. The core component of this framework is Music-Conditioned Style-Aware Diffusion (MCSAD), which comprises a Transformer-based network and a music Style Modulation module. The MCSAD seemly integrates music conditions and style description prompts into the dance generation framework, ensuring that generated dances are consistent with the music content and style. To facilitate flexible dance generation and accommodate different tasks, a spatial-temporal masking strategy is effectively applied in the backward diffusion process. The proposed framework successfully generates realistic dance sequences that are accurately aligned with music for a variety of tasks such as long-term generation, dance in-betweening, dance inpainting, and etc. We hope that this work has the potential to inspire dance generation and creation, with promising applications in entertainment, art, and education.*
## Requirements
* We follow the environment configuration of [EDGE](https://github.com/Stanford-TML/EDGE) 

## Chekpoint
* Download the saved model checkpoint from [Google Drive](https://drive.google.com/file/d/1qYEY45m3paDEXfBqpBWGCYaqvZS-Z60k/view?usp=drive_link).

## Dataset Download
Download and process the AIST++ dataset (wavs and motion only) using:
```.bash
cd data
bash download_dataset.sh
python create_dataset.py --extract-baseline --extract-jukebox
```
This will process the dataset to match the settings used in the paper. The data processing will take ~24 hrs and ~50 GB to precompute all the Jukebox features for the dataset.

## Training
Run model/style_CLIP.py to generate semantic features.
```.bash
cd model
python style_CLIP.py
```
Then, run the training script, e.g.
```.bash
cd ../
accelerate launch train.py --batch_size 128  --epochs 2000 --feature_type jukebox --learning_rate 0.0002
```
to train the model with the settings from the paper. The training will log progress to `wandb` and intermittently produce sample outputs to visualize learning.

## Testing and  Evaluation
Evaluate your model's outputs with the Beat Align Score, PFC, FID, Diversity score proposed in the paper:
1. Generate ~1k samples, saving the joint positions with the `--save_motions` argument
2. Run the evaluation script
```.bash
python test.py --music_dir custom_music/ --save_motions
python eval/beat_align_score.py
python eval/eval_pfc.py
python eval/metrics_diveristy.py
```

## Citation
```
@article{wang2024flexible,
  title={Flexible Music-Conditioned Dance Generation with Style Description Prompts},
  author={Wang, Hongsong and Zhu, Yin and Geng, Xin},
  journal={arXiv preprint arXiv:2406.07871},
  year={2024}
}
```
