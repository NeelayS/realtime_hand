from ..criterion import CriterionDSN
from ..train_seg import SegTrainer


class ESPNetTrainer(SegTrainer):
    def __init__(self, model, config_path, img_dir, bg_dir, device="cpu", **kwargs):
        super(ESPNetTrainer, self).__init__(
            model, config_path, img_dir, bg_dir, device="cpu"
        )

        self.loss_fn = CriterionDSN(**kwargs)

    def _calculate_loss(self, out, mask):

        return self.loss_fn(out, mask)
