import math

from pytorch_lightning.callbacks import TQDMProgressBar


class SplitProgressBar(TQDMProgressBar):

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        total_train_batches = self.total_train_batches

        total_batches = total_train_batches
        if total_batches is None or math.isinf(total_batches) or math.isnan(total_batches):
            total_batches = None
        if not self.main_progress_bar.disable:
            self.main_progress_bar.reset(total=total_batches)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")