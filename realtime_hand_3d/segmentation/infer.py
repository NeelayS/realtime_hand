import os

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F

from .models import SEG_MODELS_REGISTRY
from .dataset import normalize_tensor


def preprocess(img, size=(512, 288), grayscale=False, input_edge=False):

    if grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if input_edge:
        edge = cv.Canny(img.astype(np.uint8), 25, 100)
        img = np.stack((img, edge), -1)

    img = cv.resize(img, size)

    if not grayscale:

        img = img / 255.0
        IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

        img -= IMG_MEAN
        img /= IMG_VARS

        img = np.transpose(img, (2, 0, 1))

    if input_edge:
        img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    if grayscale:
        img = normalize_tensor(img, 128.0, 256.0)

    if grayscale and not input_edge:
        img = img.unsqueeze(0)

    return img


def infer_image(
    image_path, model, out_dir, device="cpu", grayscale=False, input_edge=False
):

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    img = cv.imread(image_path)
    img_r = cv.resize(img, (512, 288))
    H, W, _ = img.shape

    img = preprocess(img, size=(512, 288), grayscale=grayscale, input_edge=input_edge)

    with torch.no_grad():
        pred = model(img.to(device))

    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[0]

    pred = F.interpolate(
        pred, size=img.shape[-2:], mode="bilinear", align_corners=True
    )[0]
    pred = F.softmax(pred, dim=0)
    mask = torch.argmax(pred, dim=0).cpu().numpy()

    img_r[mask == 1] = [127, 127, 255]
    img_r[mask == 2] = [255, 127, 127]

    cv.imwrite(
        os.path.join(
            out_dir, model.__class__.__name__ + "_" + image_path.split("/")[-1]
        ),
        img_r,
    )


def infer_video(
    video_path, model, out_dir, device="cpu", grayscale=False, input_edge=False
):

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    video_name = video_path.split("/")[-1]
    cap = cv.VideoCapture(os.path.join(video_path))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out_video = cv.VideoWriter(
        os.path.join(out_dir, model.__class__.__name__ + "_" + video_name),
        fourcc,
        cap.get(cv.CAP_PROP_FPS),  # *'XVID' .avi
        (int(cap.get(3)), int(cap.get(4))),
    )

    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break

        H, W = frame.shape[:2]

        img = preprocess(frame, size=(W, H), grayscale=grayscale, input_edge=input_edge)
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)

        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

        pred = F.interpolate(
            pred, size=img.shape[-2:], mode="bilinear", align_corners=True
        )[0]
        pred = F.softmax(pred, dim=0)
        mask = torch.argmax(pred, dim=0).cpu().numpy()

        new_frame = frame
        new_frame[mask == 1] = [128, 128, 255]
        new_frame[mask == 2] = [255, 128, 128]
        new_frame = cv.resize(new_frame, (W, H))

        out_video.write(new_frame)

    cap.release()
    out_video.release()


def setup_model(model_name, weights_path=None, grayscale=False, input_edge=False):

    if grayscale:
        in_channels = 1
    if input_edge:
        in_channels = 2
    else:
        in_channels = 3

    model = SEG_MODELS_REGISTRY.get(model_name)(in_channels=in_channels, n_classes=3)

    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu")), strict=False
        )

    model.eval()

    return model


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--video", type=str, default=None, required=False)
    parser.add_argument("--image", type=str, default=None, required=False)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grayscale", action="store_true", default=False)
    parser.add_argument("--input_edge", action="store_true", default=False)
    args = parser.parse_args()

    model = setup_model(args.model, args.weights, args.grayscale, args.input_edge)

    if args.video is not None:
        infer_video(
            args.video,
            model,
            args.out_dir,
            args.device,
            args.grayscale,
            args.input_edge,
        )
    else:
        infer_image(
            args.image,
            model,
            args.out_dir,
            args.device,
            args.grayscale,
            args.input_edge,
        )
