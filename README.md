# realtime_hand_3d

<b>Real-time hand pose and shape in RGB videos</b>

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