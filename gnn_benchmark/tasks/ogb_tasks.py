from gnn_benchmark.tasks import base_tasks
import abc
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import Evaluator as GraphPropEvaluator
from ogb.nodeproppred import Evaluator as NodePropEvaluator
import ogb
import numpy as np
from gnn_benchmark.common.definitions import Channels


class OGBGraphDatasetTask(base_tasks.BaseTask, metaclass=abc.ABCMeta):
    def __init__(self, model, data_path, build_optimizer, device,
                 batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        super().__init__(model, data_path, build_optimizer, device)

        print(f"Running on task {self.task_name}")
        try:
            print(f"using OGB version {ogb.__version__}")
        except AttributeError:
            print("No OGB version found.")
        splits = self.dataset.get_idx_split()
        self.train_idx, self.test_idx, self.val_idx = splits["train"], splits["valid"], splits["test"]

    def train_dataloader(self):
        train_dataset = self.dataset[self.train_idx]
        return DataLoader(
            train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        val_dataset = self.dataset[self.val_idx]
        return [DataLoader(
            val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]

    def test_dataloader(self):
        test_dataset = self.dataset[self.test_idx]
        return [DataLoader(
            test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]

    def load_dataset(self, data_path):
        dataset = PygGraphPropPredDataset(name=self.task_name, root=data_path)
        return dataset

    def eval_on(self, loader, trainer):
        results_dict = super().eval_on(loader, trainer)
        evaluator = GraphPropEvaluator(name=self.task_name)
        y_trues = []
        y_preds = []
        for batch in loader:
            if trainer.on_gpu:
                batch = batch.to("cuda")
            y_preds.append(self.model(batch).cpu().detach().numpy())
            y_trues.append(batch.y.cpu().detach().numpy())
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
        results_dict.update(evaluator.eval({"y_true": y_trues, "y_pred": y_preds}))
        return results_dict


class BACETask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=1)

    task_name = "ogbg-mol-bace"


class BBBPTask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=1)

    task_name = "ogbg-mol-bbbp"


class CLINTOXTask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=2)
    task_name = "ogbg-mol-clintox"


class HIVTask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=1)

    task_name = "ogbg-mol-hiv"


class MUVTask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=17)
    task_name = "ogbg-mol-muv"


class PCBATask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=128)
    task_name = "ogbg-mol-pcba"


class SIDERTask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=27)
    task_name = "ogbg-mol-sider"


class TOX21Task(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=12)
    task_name = "ogbg-mol-tox21"


class TOXCASTTask(OGBGraphDatasetTask, base_tasks.GraphMultiLabelClassification):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=617)
    task_name = "ogbg-mol-toxcast"


class PPITask(OGBGraphDatasetTask, base_tasks.GraphClassificationTask):
    feature_channels = Channels(node=9, edge=3)
    target_channels = Channels(graph=37)
    task_name = "ogbg-ppi"


class OGBNodeTask(base_tasks.BaseTask, metaclass=abc.ABCMeta):
    def __init__(self, model, data_path, build_optimizer, device,
                 batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        super().__init__(model, data_path, build_optimizer, device)

        print(f"Running on task {self.task_name}")
        try:
            print(f"using OGB version {ogb.__version__}")
        except AttributeError:
            print("No OGB version found.")
        splits = self.dataset.get_idx_split()
        self.train_idx, self.test_idx, self.val_idx = splits["train"], splits["valid"], splits["test"]

    def train_dataloader(self):
        return [DataLoader(
            self.dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]


    def val_dataloader(self):
        return [DataLoader(
            self.dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]

    def test_dataloader(self):
        return [DataLoader(
            self.dataset,
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]

    def load_dataset(self, data_path):
        dataset = PygNodePropPredDataset(name=self.task_name, root=data_path)
        return dataset

    def eval_on(self, loader, trainer):
        results_dict = super().eval_on(loader, trainer)
        evaluator = NodePropEvaluator(name=self.task_name)
        y_trues = []
        y_preds = []
        for batch in loader:
            if trainer.on_gpu:
                batch = batch.to("cuda")
            y_preds.append(self.model(batch).cpu().detach().numpy())
            y_trues.append(batch.y.cpu().detach().numpy())
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
        results_dict.update(evaluator.eval({"y_true": y_trues, "y_pred": y_preds}))
        return results_dict


class ProductsTask(OGBNodeTask, base_tasks.NodeClassificationTask):
    feature_channels = Channels(node=9, edge=3)       # TODO right numbers

    target_channels = Channels(node=47)
    task_name = "ogbn-products"


class ProteinsTask(OGBNodeTask, base_tasks.NodeMultiLabelClassification):
    feature_channels = Channels(node=8, edge=8)       # TODO right numbers

    target_channels = Channels(node=112)
    task_name = "ogbn-proteins"
