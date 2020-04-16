import numpy as np
import pytorch_lightning as pl
import copy


class MemoryModelCheckpoint(pl.Callback):
    def __init__(self, monitor, mode):
        self.mode = mode
        self.monitor = monitor
        self.best_epoch = None
        self.parameters = None
        if mode == "min":
            self.comp_op = np.less
            self.best = np.inf
        elif mode == "max":
            self.comp_op = np.greater
            self.best = -np.inf
        else:
            raise ValueError(f"mode {mode} is not recognized.")

    def on_validation_end(self, trainer, pl_module):
        # only run on main process
        if trainer.proc_rank != 0:
            return

        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if self.comp_op(current, self.best):
            self.best_epoch = trainer.current_epoch
            self.best = current
            self.parameters = copy.deepcopy(pl_module.state_dict())

    def replace_parameters(self, pl_module):
        if self.parameters is not None:
            pl_module.load_state_dict(self.parameters)
        else:
            print("WARING: parameters is None; not replacing")
        return pl_module
