## Segmentation models inference times

- Tested on a V100 GPU at 512x512 input image resolution

| Model                           | Frames per second | Output resolution before interpolation (relative to input)   |
| --------------------------------| ------------------| --------------------------------------- |
| UNet1/8                         |      387.80       |                 1                       |
| SwiftNet                        |      281.57       |                 1/4                     |
| DFSegNetSmall                   |      227.49       |                 1/8                     |
| FastSCNN                        |      219.96       |                 1/8                     |
| ICNet                           |      205.55       |                 1/8                     |
| ModSegNet                       |      183.29       |                 1                       |
| DFSegNet                        |      165.71       |                 1/8                     |
| CSM                             |      115.6        |                 1/4                     |
| SegNet                          |      68.43        |                 1                       |
| UNet                            |      65.52        |                 1                       |
| LightWeightRefineNet            |      50.90        |                 1/4                     |
| PSPNet                          |      28.32        |                 1/8                     |
| RefineNet                       |      28.05        |                 1/4                     |



## Hand Segmentation Datasets 

<b> Table summarizing important info about the prominent datasets </b>

| Name         | No. of hands  | No. of frames | View     | Dataset link | Paper link | 
|--------------|---------------|---------------|----------|-------------------|-------|
| Ego2Hands | 2 | 188,362 | Egocentric | [Dataset](https://github.com/AlextheEngineer/Ego2Hands) | [Paper](https://arxiv.org/abs/2011.07252) |
| EgoHands | 1-4 | 4,800 | Egocentric | [Dataset](http://vision.soic.indiana.edu/projects/egohands/) | [Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Bambach_Lending_A_Hand_ICCV_2015_paper.html) |
| EGTEA | 1-2 (1 in most frames) | 13,847 | Egocentric | [Dataset](http://cbs.ic.gatech.edu/fpv/) | [Paper](https://arxiv.org/abs/2006.00626) |
| EgoYouTubeHands | 1-2 | 1,290 | Egocentric | [Dataset](https://github.com/aurooj/Hand-Segmentation-in-the-Wild) | [Paper](https://arxiv.org/abs/1803.03317) |
| HandOverFace | 1-2 | 300 | Third-person | [Dataset](https://github.com/aurooj/Hand-Segmentation-in-the-Wild) | [Paper](https://arxiv.org/abs/1803.03317) |

<b> Table extensively comparing all hand segmentation datasets </b> <br>
- From the Ego2Hands paper

![Table](./datasets_comparison.png "Datasets comparison table")