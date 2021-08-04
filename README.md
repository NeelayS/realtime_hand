# realtime_hand_3d

<b>Real-time hand pose and shape in RGB videos</b>

<br>

## Steps to train 

- Create conda env using the `env.yml` file.

```
conda env create -f env.yml
conda activate hand
```

- Install relevant version of PyTorch by substituting appropriate CUDA version in the following command -

```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Download training data by -

```
cd segmentation/data
bash download_data.sh
```

This will download the compressed data files, extract them to `segmentation/data/train_imgs` and `segmentation/data/bg_imgs`, and will then delete the compressed files.

- If the above command doesn't work out -

    - Download files from [1](https://byu.box.com/s/moy2j92p9j9tv8mw8c1dgafn4r4pod19), [2](https://byu.box.com/s/jdto18tt4q89pdmn2l2wiiics2ltdr54), [3](https://byu.box.com/s/0yj1iqlsmt7aw7odp3ns50e39nmer4vo) and [4](https://byu.box.com/s/fr3lcjscu5xit6qbyqdooy6pi6uyk1q3) and extract them to `segmentation/data/train_imgs`.
    - Download file from [1](https://byu.box.com/s/dc16feb1nhswm3imtce7f6r5ai7d0i6w) and extract to `segmentation/data/bg_imgs`.

- To train, switch to the `scripts` directory and run `train_segmentation.sh` with the command line arguments described below -
    1. model_name: Name of the segmentation model to train, eg. LightWeightRefineNet
    2. device: CPU/GPU Cud device to train on.
        - Enter '-1' to use CPU
        - Enter '0' to use CUDA device 0
        - Enter '0,1' to use CUDA devices 0 and 1 (no spaces)
        - Enter 'all' to use all CUDA devices available

    Example command `bash train_segmentation.sh LightWeightRefineNet 0`

- To view training curves after completion -

```
tensorboard --logdir=experiments
```

<br>

## Model inference times

- Tested on Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz

| Model                           | Frames per second |
| --------------------------------| ------------------|
| UNet                            |      3.65         |
| UNet1/8                         |      54.45        |
| SegNet                          |      3.35         |
| ModSegNet                       |      10.43        |
| RefineNet                       |      1.08         |
| LightWeightRefineNet            |      1.67         |
| ICNet                           |      3.90         |

<br>

## Current directory structure - 

```
/
├── generate_dir_structure.py
└── segmentation/
    ├── models/
    │   ├── icnet/
    │   │   ├── blocks.py
    │   │   └── icnet.py
    │   ├── refinenet/
    │   │   ├── blocks.py
    │   │   ├── lightweight_refinenet.py
    │   │   └── refinenet.py
    │   ├── segnet/
    │   │   ├── modsegnet.py
    │   │   └── segnet.py
    │   ├── test_models.py
    │   └── unet/
    │       ├── custom_unet.py
    │       └── unet.py
    └── test_inference.py
```

### Immediate ToDos -

- [ ] Add implementations of
    - [x] UNet
    - [x] Unet1/8
    - [x] SegNet
    - [x] RefineNet
    - [X] ICNet
    - [ ] DeepLabV3+
- [ ] Add inference speed and accuracy tests for all models
- [ ] Set up code tests using GitHub actions/Travis CI 
- [ ] Add load pretrained model method for RefineNet


### Gradual ToDos - 

- Ensure adherence to licenses of all open source code references