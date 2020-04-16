from gnn_benchmark.tasks import base_tasks
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric import transforms as geom_transform
from gnn_benchmark.common import transforms
from gnn_benchmark.common.definitions import Channels


class CitationTask(base_tasks.NodeClassificationTask):
    def __init__(self, model, data_path, build_optimizer, device,
                 batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        super().__init__(model, data_path, build_optimizer, device)

        print(f"Running on task {self.task_name}")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return [DataLoader(
            self.dataset["val"],
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]

    def test_dataloader(self):
        return [DataLoader(
            self.dataset["test"],
            self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )]

    def load_dataset(self, data_path):
        datasets = {}
        for set_name in ["train", "val", "test"]:
            dataset = Planetoid(
                root=data_path, name=self.task_name
            )
            dataset.transform = geom_transform.Compose([
                geom_transform.NormalizeFeatures(),
                transforms.CreatePlanetoidSplit(self.target_channels.node, seed=0, mask_name=f"{set_name}_mask",
                                                randomize=False)
            ])
            datasets[set_name] = dataset
        return datasets


class CiteseerTask(CitationTask):
    task_name = "CiteSeer"
    feature_channels = Channels(node=3703)
    target_channels = Channels(node=6)


class CoraTask(CitationTask):
    task_name = "Cora"
    feature_channels = Channels(node=1433)
    target_channels = Channels(node=7)


class PubmedTask(CitationTask):
    task_name = "PubMed"
    feature_channels = Channels(node=500)
    target_channels = Channels(node=3)

