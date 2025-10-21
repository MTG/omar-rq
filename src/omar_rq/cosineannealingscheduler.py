import math
import gin.torch
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, eta_min, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

        assert self.total_steps > self.warmup_steps, (
            f"total_steps: {self.total_steps} must be greater than warmup_steps: {self.warmup_steps}"
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr * (self.last_epoch + 1) / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


@gin.configurable
class CosineAnnealingCallback(Callback):
    def __init__(self, total_steps, warmup_steps, eta_min):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min

    def on_train_start(self, trainer, pl_module):
        last_epoch = -1
        # If resuming from a checkpoint, set last_epoch to the global_step
        if trainer.global_step > 0:
            last_epoch = trainer.global_step

        optimizer = trainer.optimizers[0]
        self.scheduler = CosineAnnealingWithWarmup(
            optimizer,
            self.total_steps,
            self.warmup_steps,
            self.eta_min,
            last_epoch=last_epoch,
        )

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.scheduler.step()
        pl_module.log("lr", self.scheduler.get_last_lr()[0])
