"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

from torch.optim import SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


class Registry:
    def __init__(self, name):

        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):

        assert (
            name not in self._obj_map
        ), f"An object named '{name}' was already registered in '{self._name}' registry!"

        self._obj_map[name] = obj

    def register(self, obj=None, name=None):

        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        if name is None:
            name = obj.__name__

        self._do_register(name, obj)

    def get(self, name):

        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )

        return ret

    def get_list(self):
        return list(self._obj_map.keys())

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())


optimizers = Registry("optimizers")
schedulers = Registry("schedulers")


optimizers.register(SGD, "SGD")
optimizers.register(Adam, "Adam")
optimizers.register(AdamW, "AdamW")
optimizers.register(Adagrad, "Adagrad")
optimizers.register(Adadelta, "Adadelta")
optimizers.register(RMSprop, "RMSprop")

schedulers.register(CosineAnnealingLR, "CosineAnnealingLR")
schedulers.register(CosineAnnealingWarmRestarts, "CosineAnnealingWarmRestarts")
schedulers.register(CyclicLR, "CyclicLR")
schedulers.register(MultiStepLR, "MultiStepLR")
schedulers.register(ReduceLROnPlateau, "ReduceLROnPlateau")
schedulers.register(StepLR, "StepLR")
schedulers.register(OneCycleLR, "OneCycleLR")
