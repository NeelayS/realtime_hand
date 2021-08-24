from ..criterion import CriterionDSN
from ...train_seg import SegTrainer


class DFSegNetTrainer(SegTrainer):
    def __init__(self, model, config_path, img_dir, bg_dir, device="cpu", **kwargs):
        super(DFSegNetTrainer, self).__init__(
            model, config_path, img_dir, bg_dir, device
        )

        self.loss_fn = CriterionDSN(**kwargs)

    def _calculate_loss(self, out, mask):

        return self.loss_fn(out, mask)
