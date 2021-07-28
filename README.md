# realtime_hand_3d

<b>Real-time hand pose and shape in RGB videos</b>

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