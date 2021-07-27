import os
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from time import time

from .utils import preprocessing, draw_fore_to_back, draw_matting


def test_inference(
    video_path, model, viz=False, out_dir=None, device="cpu", inp_size=320
):

    video_name = video_path.split("/")[-1]
    cap = cv.VideoCapture(os.path.join(video_path))

    if viz:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out_video = cv.VideoWriter(
            os.path.join(out_dir, video_name),
            fourcc,
            cap.get(cv.CAP_PROP_FPS),
            (int(cap.get(3)), int(cap.get(4))),
        )

    model = model.to(device)
    model = model.eval()

    inference_times = []

    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break

        image = frame[..., ::-1]
        H, W = image.shape[:2]

        if H==W:
            inp_size=H

        X, pad_up, pad_left, h_new, w_new = preprocessing(
            image, expected_size=inp_size, pad_value=0
        )

        with torch.no_grad():

            X = X.to(device)

            infer_start = time()

            mask = model(X)

            infer_end = time()

            mask = mask[..., pad_up : pad_up + h_new, pad_left : pad_left + w_new]
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=True)
            mask = F.softmax(mask, dim=1)
            mask = mask[0, 1, ...].cpu().numpy()

        image_alpha = draw_matting(image, mask)

        inference_times.append(infer_end - infer_start)

        if viz:
            out_video.write(image_alpha)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    fps = 1 / (sum(inference_times) / len(inference_times))
    print(f"The model performs inference at {fps} fps")

    if viz:
        out_video.release()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    from argparse import ArgumentParser
    from .models import (
        UNet,
        CustomUNet,
        SmallUNet,
        CustomSmallUNet,
        ICNet,
        RefineNet,
        SegNet,
        ModSegNet,
    )

    parser = ArgumentParser("Utility to measure inference time of models")
    parser.add_argument(
        "--video", type=str, required=True, help="Video to be used for inference"
    )
    parser.add_argument("--model", type=str, required=True, help="Model to be used")
    parser.add_argument(
        "--viz",
        type=bool,
        default=False,
        help="Whether to visualize model performance and save video",
    )
    parser.add_argument(
        "--out_dir", type=str, default=".", help="Directory to save output video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="CPU/GPU device to be used for inference",
    )
    parser.add_argument(
        "--inp_size", type=int, default=320, help="Input image size the model expects"
    )
    args = parser.parse_args()

    models = {
        "unet": UNet,
        "customunet": CustomUNet,
        "smallunet": SmallUNet,
        "customsmallunet": CustomSmallUNet,
        "icnet": ICNet,
        "refinenet": RefineNet,
        "segnet": SegNet,
        "modsegnet": ModSegNet,
    }

    model = models[args.model](n_classes=2)

    test_inference(
        args.video, model, args.viz, args.out_dir, args.device, args.inp_size
    )
