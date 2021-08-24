from ...train_seg import SegTrainer
from .criterion import ModCriterionICNet, CriterionICNet


class ICNetTrainer(SegTrainer):
    def __init__(self, model, config_path, img_dir, bg_dir, device="cpu", **kwargs):
        super(ICNetTrainer, self).__init__(
            model, config_path, img_dir, bg_dir, device="cpu"
        )

        self.loss_fn = ModCriterionICNet(**kwargs)

    def _calculate_loss(self, out, mask):

        return self.loss_fn(out, mask)


class CustomICNetTrainer(SegTrainer):
    def __init__(self, model, config_path, img_dir, bg_dir, device="cpu", **kwargs):
        super(CustomICNetTrainer, self).__init__(
            model, config_path, img_dir, bg_dir, device="cpu"
        )

        self.loss_fn = CriterionICNet(**kwargs)

    def _calculate_loss(self, out, mask):

        return self.loss_fn(out, mask)
