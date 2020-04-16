from gnn_benchmark.tasks.base_tasks import BaseTask, GraphClassificationTask
from torch_geometric.datasets import TUDataset as geomTUDataset
import abc
from torch_geometric.data import DataLoader
from gnn_benchmark.common.utils import k_fold
from gnn_benchmark.common.transforms import create_one_hot_transform, create_categorical_degree_transform
from gnn_benchmark.common.definitions import Channels

class TUDatasetTask(BaseTask, metaclass=abc.ABCMeta):
    features = None
    targets = None

    def __init__(self, model, data_path, build_optimizer, device, folds, fold_idx,
                 batch_size, num_workers, cleaned=False):
        self.cleaned = cleaned
        self.folds = folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().__init__(model, data_path, build_optimizer, device)
        self.data_path = data_path
        self.train_idx, self.test_idx, self.val_idx = list(
            zip(*k_fold(self.dataset, self.folds))
        )[self.fold_idx]

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

    @abc.abstractmethod
    def load_dataset(self, data_path):
        pass


class MUTAGTask(TUDatasetTask, GraphClassificationTask):
    feature_channels = Channels(node=7, edge=4)
    target_channels = Channels(graph=2)

    def load_dataset(self, data_path):
        dataset = geomTUDataset(
            root=data_path, name="MUTAG", cleaned=self.cleaned
        )
        return dataset


class PROTEINSTask(TUDatasetTask, GraphClassificationTask):
    feature_channels = Channels(node=3)
    target_channels = Channels(graph=2)

    def load_dataset(self, data_path):
        dataset = geomTUDataset(
            root=data_path, name="PROTEINS", cleaned=self.cleaned
        )
        return dataset


class IMDBBinaryTask(TUDatasetTask, GraphClassificationTask):
    feature_channels = Channels(node=136)
    target_channels = Channels(graph=2)

    def load_dataset(self, data_path):
        dataset = geomTUDataset(
            root=data_path, name="IMDB-BINARY", cleaned=self.cleaned
        )
        dataset.transform = create_one_hot_transform(dataset)
        return dataset


class REDDITBinaryTask(TUDatasetTask, GraphClassificationTask):
    feature_channels = Channels(node=1)
    target_channels = Channels(graph=2)

    def load_dataset(self, data_path):
        dataset = geomTUDataset(
            root=data_path, name="REDDIT-BINARY", cleaned=self.cleaned
        )
        dataset.transform = create_categorical_degree_transform(dataset)
        return dataset


class COLLABTask(TUDatasetTask, GraphClassificationTask):
    feature_channels = Channels(node=492)
    target_channels = Channels(graph=3)

    def load_dataset(self, data_path):
        dataset = geomTUDataset(
            root=data_path, name="COLLAB", cleaned=self.cleaned
        )
        dataset.transform = create_one_hot_transform(dataset)
        return dataset


# ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']
