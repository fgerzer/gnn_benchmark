from dataclasses import dataclass, field
import numpy as np


@dataclass
class Channels:
    node: int = None
    edge: int = None
    graph: int = None


@dataclass
class RunResults:
    duration: float = None
    trainable_parameters: int = None
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    gpu_mem_usage: int = None

    def to_dict(self):
        return {
            "duration": self.duration,
            "trainable_parameters": self.trainable_parameters,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
            "gpu_mem_usage": self.gpu_mem_usage
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __str__(self):
        with np.printoptions(precision=3, suppress=True):
            return f"Results: duration: {self.duration:.2f}s, trainable parameters: {self.trainable_parameters}, " \
                   f"GPU memory usage: {self.gpu_mem_usage // (1024**2) if self.gpu_mem_usage is not None else None}MB\n" \
               f"\ttrain: {self.train_metrics}\n\tval: {self.val_metrics}\n\ttest: {self.test_metrics}"


class RunState:
    running = "running"
    pending = "pending"
    finished = "finished"


@dataclass
class RunEntry:
    run_state: str
    run_definition: dict
    results: RunResults
    id: str or None

    def to_dict(self):
        return {
            "run_state": self.run_state,
            "run_definition": self.run_definition,
            "results": self.results.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            run_state=d["run_state"],
            run_definition=d["run_definition"],
            results=RunResults.from_dict(d["results"]),
            id=d["_id"]
        )
