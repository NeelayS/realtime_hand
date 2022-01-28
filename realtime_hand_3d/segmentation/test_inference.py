import os
from time import time

import cv2 as cv
import torch
import torch.nn.functional as F

from .utils import draw_fore_to_back, draw_matting, preprocessing


def warmup(model, device, inp_size=512):

    inp = torch.rand(1, 3, inp_size, inp_size).to(device)
    model(inp)


def test_seg_inference(
    video_path, model, viz=False, out_dir=None, device="cpu", inp_size=512
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

    model = model.to(torch.device(device))
    model = model.eval()

    warmup(model, device, inp_size)

    inference_times = []

    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break

        image = frame[..., ::-1]
        H, W = image.shape[:2]

        if H == W:
            inp_size = H

        X, pad_up, pad_left, h_new, w_new = preprocessing(
            image, expected_size=inp_size, pad_value=0
        )

        with torch.no_grad():

            X = X.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            infer_start = time()

            pred = model(X)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            infer_end = time()
            inference_times.append(infer_end - infer_start)

            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[0]

            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
            mask = F.softmax(pred, dim=1)

            # mask = mask[..., pad_up : pad_up + h_new, pad_left : pad_left + w_new]

            assert mask.shape[-2:] == (H, W), "Prediction shape is not correct"

            # mask = mask[0, 1, ...].cpu().numpy()

        # image_alpha = draw_matting(image, mask)

        # if viz:
        #     out_video.write(image_alpha)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    avg_inference_time = sum(inference_times) / len(inference_times)
    fps = 1 / avg_inference_time
    print(f"{model.__class__.__name__} performs inference at {fps} fps")

    if viz:
        out_video.release()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    from argparse import ArgumentParser

    from .models import SEG_MODELS_REGISTRY

    parser = ArgumentParser("Utility to measure inference time of models")
    parser.add_argument(
        "--video", type=str, required=True, help="Video to be used for inference"
    )
    parser.add_argument("--model", type=str, required=False, help="Model to be used")
    parser.add_argument(
        "--all_models",
        action="store_true",
        default=False,
        help="Whether to test inference for all models",
    )
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

    if args.model is None and args.all_models is False:
        raise Exception(
            "Please specify either a model name or set the 'all_models' flag to True"
        )

    if args.model:

        assert args.model in SEG_MODELS_REGISTRY, "Model not found in registry"
        model = SEG_MODELS_REGISTRY.get(args.model)(in_channels=3, n_classes=3)

        test_seg_inference(
            args.video, model, args.viz, args.out_dir, args.device, args.inp_size
        )

    else:

        for model_name in sorted(SEG_MODELS_REGISTRY.get_list()):

            if model_name in ("BiSegNet", "ESPNet", "DFANet", "CustomICNet"):
                print(f"\nSkipping {model_name}")
                continue

            print(f"\nTesting inference for {model_name}")

            try:
                model = SEG_MODELS_REGISTRY.get(model_name)(in_channels=3, n_classes=3)
            except:
                print(f"{model_name} failed to load")
                continue

            try:
                inp = torch.randn(2, 3, args.inp_size, args.inp_size)
                # print(inp.shape)
                _ = model(inp)
            except:
                print(f"\n{model_name} doesn't work for the specified input size")
                continue

            test_seg_inference(
                args.video, model, args.viz, args.out_dir, args.device, args.inp_size
            )
