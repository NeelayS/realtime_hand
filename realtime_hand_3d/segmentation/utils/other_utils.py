import yaml
from easydict import EasyDict


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Config(filename):

    with open(filename, "r") as f:
        parser = EasyDict(yaml.safe_load(f))

    # for x in parser:
    #     print(f"{x}: {parser[x]}")

    return parser


def normalize_tensor(tensor, mean, std):

    for t in tensor:
        t.sub_(mean).div_(std)

    return tensor
