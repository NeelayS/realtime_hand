import yaml
from easydict import EasyDict


def Config(filename):

    with open(filename, "r") as f:
        parser = EasyDict(yaml.safe_load(f))

    # for x in parser:
    #     print(f"{x}: {parser[x]}")

    return parser
