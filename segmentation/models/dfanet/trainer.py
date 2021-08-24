from ...train_seg import SegTrainer
from .criterion import CriterionDFANet


class DFANetTrainer(SegTrainer):
    def __init__(self, model, config_path, img_dir, bg_dir, device="cpu", **kwargs):
        super(DFANetTrainer, self).__init__(
            model, config_path, img_dir, bg_dir, device="cpu"
        )

        self.loss_fn = CriterionDFANet(**kwargs)

    def _calculate_loss(self, out, mask):

        return self.loss_fn(out, mask)
