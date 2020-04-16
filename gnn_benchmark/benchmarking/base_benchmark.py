import abc
from gnn_benchmark.common.run_db import RunsDB, RunState
from gnn_benchmark.common import definitions as defs
import torch
import pytorch_lightning as pl
import time
from gnn_benchmark.common.utils import num_trainable
from gnn_benchmark.common.callbacks import MemoryModelCheckpoint


class BaseBenchmark(metaclass=abc.ABCMeta):
    def __init__(self, db_paths, data_path):
        print(f"Using database {db_paths.host}\t{db_paths.database}\t{db_paths.collection}")
        self.runs_db = RunsDB(db_paths)
        self.data_path = data_path

    @abc.abstractmethod
    def create_params(self):
        pass

    @abc.abstractmethod
    def exec_single_run(self, **kwargs):
        pass

    def create_runs(self):
        if self.runs_db.lock():
            try:
                param_configs = self.create_params()
                runs = self.runs_db.insert_runs(param_configs)
                return runs
            finally:
                self.runs_db.unlock()
        else:
            self.runs_db.wait_for_unlock()

    def worker_loop(self):
        trial = self.run_single()
        while trial is not None:
            trial = self.run_single()

    def run_single(self):
        trial = self.runs_db.claim_run()
        if trial is None:
            return None
        n_runs = self.runs_db.n_runs()
        n_pending = self.runs_db.n_runs(run_state=RunState.pending)
        print(f"Running task {trial.id}. {n_pending}/{n_runs} remaining ({100 * n_pending // n_runs}%)")
        print(trial.run_definition)
        run_results = self.exec_single_run(**trial.run_definition)
        print(run_results)
        print("\n\n------------------------------------")
        self.runs_db.submit_result(trial, run_results)
        return trial

    def run_and_eval(self, task, max_epochs, gpus, early_stop_patience=None):
        checkpoint_callback = MemoryModelCheckpoint(
            monitor="loss",
            mode="min"
        )
        if early_stop_patience is None:
            early_stop_callback = False
        else:
            early_stop_callback = pl.callbacks.EarlyStopping(monitor="loss", patience=early_stop_patience, mode="min")
        trainer = pl.Trainer(
            checkpoint_callback=False,
            logger=False,
            show_progress_bar=False,
            max_epochs=max_epochs,
            early_stop_callback=early_stop_callback,
            gpus=gpus,
            weights_summary=None,
            callbacks=[checkpoint_callback]
        )
        run_results = defs.RunResults(trainable_parameters=num_trainable(task))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
        t_start = time.perf_counter()
        trainer.fit(task)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_usage = torch.cuda.max_memory_allocated()
            run_results.gpu_mem_usage = mem_usage
        t_end = time.perf_counter()

        duration = t_end - t_start
        run_results.duration = duration
        print(f"Loading best checkpoint from epoch {checkpoint_callback.best_epoch}")
        task = checkpoint_callback.replace_parameters(pl_module=task)

        run_results.train_metrics = task.eval_on(task.train_dataloader(), trainer)
        run_results.val_metrics = task.eval_on(task.val_dataloader()[0], trainer)
        run_results.test_metrics = task.eval_on(task.test_dataloader()[0], trainer)
        return run_results
