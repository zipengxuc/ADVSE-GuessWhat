Answer-Driven Visual State Estimator for Goal-Oriented Visual Dialogue
====================================

This repository contains the code for the following paper:

* Zipeng Xu, Fangxiang Feng, Xiaojie Wang, Yushu Yang, Huixing Jiang, Zhongyuan Wang, *Answer-Driven Visual State Estimator for Goal-Oriented Visual Dialogue*. In ACM MM, 2020. ([PDF](https://arxiv.org/pdf/2010.00361.pdf))

```
@inproceedings{10.1145/3394171.3413668,
author = {Xu, Zipeng and Feng, Fangxiang and Wang, Xiaojie and Yang, Yushu and Jiang, Huixing and Wang, Zhongyuan},
title = {Answer-Driven Visual State Estimator for Goal-Oriented Visual Dialogue},
year = {2020},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {4271–4279}
}
```

This code is reimplemented as a fork of [GuessWhatGame/guesswhat](https://github.com/GuessWhatGame/guesswhat), which uses tensorflow.

### Data
GuessWhat?! relies on two datasets:
 - the [GuessWhat?!](https://guesswhat.ai/) dataset that contains the dialogue inputs
 - The [MS Coco](http://mscoco.org/) dataset that contains the image inputs

### Supervised Pretraining

Train Oracle:
```
python src/guesswhat/train/train_oracle.py \
   -config config/oracle/config.v1.json \
   -exp_dir out/oracle
```
Train ADVSE-QGen:
```
python src/guesswhat/train/train_qgen_supervised.py \
   -config config/qgen/config.advse.json \
   -exp_dir out/qgen 
```
Train ADVSE-Guesser:
```
python src/guesswhat/train/train_guesser.py \
   -config config/guesser/config.advse.json \
   -exp_dir out/guesser 
```

### Reinforcement Learning

Based on the supervisedly pretrained models, we use REINFORCE to fine-tune the QGen model.
```
python src/guesswhat/train/train_qgen_reinforce.py
    -exp_dir out/loop/ \
    -config config/looper/config.advse8g.json \
    -networks_dir out/ \
    -oracle_identifier <oracle_identifier> \
    -qgen_identifier <qgen_identifier> \
    -guesser_identifier <guesser_identifier> \
    -evaluate_all false \
    -store_games true 
```
