from pathlib import Path

from utils.reproducibility import make_reproducible
from utils.lightning import SplitProgressBar


from configs import Config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = Config.General.visible_dev
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model import PCRNetwork as Model
import pytorch_lightning as pl
import wandb


def main():
    model = Model(Config.Model)

    project = 'pcr-grasping'

    ckpt_path = None
    loggers = []

    if Config.Eval.wandb:
        wandb.init(project=project, reinit=True, config=Config.to_dict(),
                   settings=wandb.Settings(start_method="thread"))
        wandb_logger = WandbLogger(log_model='all')

        wandb_logger.watch(model, log='all', log_freq=Config.Eval.log_metrics_every)
        loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_models/jaccard',
        dirpath='checkpoint',
        filename='epoch{epoch:02d}-cd{val_models/jaccard:.2f}',
        mode='min',
        save_last=True,
        auto_insert_metric_name=False)

    trainer = pl.Trainer(max_epochs=Config.Train.n_epoch,
                         precision=32,
                         gpus=1,
                         log_every_n_steps=10,
                         check_val_every_n_epoch=Config.Eval.val_every,
                         logger=loggers,
                         gradient_clip_val=Config.Train.clip_value,
                         gradient_clip_algorithm='value',
                         num_sanity_val_steps=4,
                         callbacks=[
                             checkpoint_callback,
                             SplitProgressBar()],
                         )

    trainer.fit(model, ckpt_path=ckpt_path)
