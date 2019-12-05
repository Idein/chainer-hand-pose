# How to prepare dataset for training

In this document, we will explain how to prepare dataset for training/inference
for our purpose. We will assume you will gather these datset under `~/dataset` directory.
Note that you do not have to collect all dataset. If you would like to try our script as soon as possible,
we recommend download `RHD Dataset` and `Stereo Hand Pose Tracking Benchmark` at least.

# RGB Dataset

## First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations (FHAD)

1. Go to [web site](https://guiggh.github.io/publications/first-person-hands/)
and press button `Dataset download` to get permission to download their dataset.
1. Prepare dataset directory, for example, `DATASET_DIR=~/dataset/fhad`.
1. Get `Hand_pose_annotation_v1.zip` and unzip in `DATASET_DIR`
1. Get `Video_files/Subject_X.zip` (for X=1, 2, 3, 4, 5, 6) and unzip them in `DATASET_DIR`

  - By following the download procedure above, you'll get the following directory structure:
```
$ tree -d -L 2 ~/dataset/fhad
.
├── Hand_pose_annotation_v1
│   ├── Subject_1
│   ├── Subject_2
│   ├── Subject_3
│   ├── Subject_4
│   ├── Subject_5
│   └── Subject_6
└── Video_files
    ├── Subject_1
    ├── Subject_2
    ├── Subject_3
    ├── Subject_4
    ├── Subject_5
    └── Subject_6
```


## Stereo Hand Pose Tracking Benchmark (a.k.a STB)
1. Prepare dataset directory, for example, `DATASET_DIR=~/dataset/stb`
1. Go to repository [zhjwustc/icip17_stereo_hand_pose_dataset](https://github.com/zhjwustc/icip17_stereo_hand_pose_dataset) and click link of `Dropbox` i.e.
  - `https://www.dropbox.com/sh/ve1yoar9fwrusz0/AAAfu7Fo4NqUB7Dn9AiN8pCca?dl=0`
1. Download all files e.g. `BXCounting.zip`, `BXRandom.zip` (for X = 1,2,3,4,5,6) or `labels.zip` and unzip them.
  - Note that since some folder doesn't generate directory, you'll need add `-d` option.
```
# for X = 1,2,3
$ unzip BXCounting.zip -d BXCounting
$ unzip BXRandom.zip -d BXRandom
# for X = 4,5,6
$ unzip BXCounting.zip
$ unzip BXRandom.zip
# finally unzip labels.zip
$ unzip labels.zip
```
  - By following the download procedure above, you'll get the following directory structure.
```
$ tree -d -L 2 ~/dataset/stb
.
├── images
│   ├── B1Counting
│   ├── B1Random
│   ├── B2Counting
│   ├── B2Random
│   ├── B3Counting
│   ├── B3Random
│   ├── B4Counting
│   ├── B4Random
│   ├── B5Counting
│   ├── B5Random
│   ├── B6Counting
│   └── B6Random
└── labels
```


## GANerated Hands Dataset

1. Prepare dataset directory, for example, `~/dataset`
1. Go to website https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm and Download `GANerated Hands Data:
Compressed Zip: Single file (zip, 33.7 GB)`. You can get `GANeratedDataset_v3.zip`
  - Extract `GANeratedDataset_v3.zip`
  ```
  $ cd ~/dataset
  $ unzip GANeratedDataset_v3.zip
  $ ls
  GANeratedDataset_v3.zip GANeratedHands_Release
  ```
  - By following the download procedure above, you'll get the following directory structure.
```console
$ tree -d -L 2 ~/dataset/GANeratedDataset_Release
.
└── data
    ├── noObject
    └── withObject
```

## SynthHands

1. Prepare dataset directory, for example, `~/dataset`
1. Go to website https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/SynthHands.htm and Download `Compressed Zip: Single file (zip, 48.5 GB)`. You can get `SynthHands.zip`.
  - Extract `SynthHands.zip`:
  ```
  $ cd ~/dataset
  $ unzip SynthHands.zip
  $ ls
  SynthHands.zip SynthHands_Release
  ```
  - By following the download procedure above, you'll get the following directory structure.
```
$ tree -d -L 2 ~/dataset/SynthHands_Release
├── female_noobject
│   ├── seq01
│   ├── seq02
│   ├── seq03
│   ├── seq04
│   ├── seq05
│   ├── seq06
│   └── seq07
├── female_object
│   ├── seq01
│   ├── seq02
│   ├── seq03
│   ├── seq04
│   ├── seq05
│   ├── seq06
│   └── seq07
├── male_noobject
│   ├── seq01
│   ├── seq02
│   ├── seq03
│   ├── seq04
│   ├── seq05
│   ├── seq06
│   └── seq07
└── male_object
    ├── seq01
    ├── seq02
    ├── seq03
    ├── seq04
    ├── seq05
    ├── seq06
    └── seq07
```


## RHD dataset

Go to [website](https://lmb.informatik.uni-freiburg.de/projects/hand3d/) and download dataset and unzip it.

```
$ unzip RHD_v1-1.zip
# you will see `RHD_published_v2` directory
```

You can see the following directory structure

```
$ tree -d -L 2 RHD_published_v2
RHD_published_v2/
├── evaluation
│   ├── color
│   ├── depth
│   └── mask
└── training
    ├── color
    ├── depth
    └── mask
```


## Large-scale Multiview 3D Hand Pose Dataset

go to [website](http://www.rovit.ua.es/dataset/mhpdataset/) and download dataset (see at the bottom of the website). You can get `multiview_hand_pose_dataset_uploaded_v2.zip`. Extract it and rename it as `multiview_hand`

You can confirm the directory structure as follow:

```
tree -d -L 2 multiview_hand
multiview_hand
├── annotated_frames
│   ├── data_1
│   ├── data_10
│   ├── data_11
│   ├── data_12
│   ├── data_13
│   ├── data_14
│   ├── data_15
│   ├── data_16
│   ├── data_17
│   ├── data_18
│   ├── data_19
│   ├── data_2
│   ├── data_20
│   ├── data_21
│   ├── data_3
│   ├── data_4
│   ├── data_5
│   ├── data_6
│   ├── data_7
│   ├── data_8
│   └── data_9
├── augmented_samples
│   ├── data_1
│   ├── data_11
│   ├── data_14
│   ├── data_15
│   ├── data_16
│   ├── data_17
│   ├── data_18
│   ├── data_19
│   ├── data_2
│   ├── data_21
│   ├── data_3
│   ├── data_4
│   ├── data_5
│   ├── data_6
│   ├── data_7
│   ├── data_8
│   └── data_9
├── calibrations
│   ├── data_1
│   ├── data_10
│   ├── data_11
│   ├── data_12
│   ├── data_13
│   ├── data_14
│   ├── data_15
│   ├── data_16
│   ├── data_17
│   ├── data_18
│   ├── data_19
│   ├── data_2
│   ├── data_20
│   ├── data_21
│   ├── data_3
│   ├── data_4
│   ├── data_5
│   ├── data_6
│   ├── data_7
│   ├── data_8
│   └── data_9
└── utils
```


## Hand Keypoint Detection in Single Images using Multiview Bootstrapping
1. Make directory named `handdb_dataset` and
1. Go to [website](http://domedb.perception.cs.cmu.edu/handdb.html) and download them at `handdb_dataset`:
  - `Hands with Manual Keypoint Annotations`
  - `Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)`
  - `Hands from Panoptic Studio by Multiview Bootstrapping (14817 annotations)`
1. Move them at `handdb_dataset` and extract them.
1. You will see the following structure:

```
$ tree -d -L 2 ~/dataset/handdb_dataset
handdb_dataset/
├── hand143_panopticdb
│   └── imgs
├── hand_labels
│   ├── manual_test
│   ├── manual_train
│   └── output_viz
└── hand_labels_synth
    ├── output_viz_synth
    ├── synth1
    ├── synth2
    ├── synth3
    └── synth4
```


## FreiHAND Dataset

1. Go to [website](https://lmb.informatik.uni-freiburg.de/projects/freihand/) and download dataset FreiHand Dataset.
1. You will get `FreiHAND_pub_v1.zip` and extract it.
1. You will see the following structure:

```
tree -L 2 -d FreiHAND_pub_v1
FreiHAND_pub_v1
├── evaluation
│   └── rgb
└── training
    ├── mask
    └── rgb
```

# Depth Dataset

## NYU Hand pose dataset

1. Prepare dataset directory, for example, DATASET_DIR=~/dataset/nyu_hand_dataset_v2
1. Go to website https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm, download `nyu_hand_dataset_v2.zip (92 GB)`
  - Extract `nyu_hand_dataset_v2.zip` with the following command:
```console
$ cd ~/dataset/nyu_hand_dataset_v2
$ unzip nyu_hand_dataset_v2.zip -d nyu_hand_dataset_v2
```

  - By following the download procedure above, you'll get the following directory structure.
```console
$ tree -d -L 2 ~/dataset/nyu_hand_datset_v2
.
└── dataset
    ├── test
    └── train
```

1. Finally, to specify this dataset for training or inference procedure, edit `confing.ini` to set `nyu` to `type` option of `dataset` section.

## 2015 MSRA Hand Gesture Dataset

1. Go to website https://jimmysuen.github.io/ and download `2015 MSRA Hand Gesture Dataset`.
1. You will get `cvpr15_MSRAHandGestureDB.zip` and extract it.
  - By following the download procedure above, you'll get the following directory structure.
```console
$ tree -d -L 1 ~/dataset/cvpr15_MSRAHandGestureDB
cvpr15_MSRAHandGestureDB
├── P0
├── P1
├── P2
├── P3
├── P4
├── P5
├── P6
├── P7
└── P8
```
