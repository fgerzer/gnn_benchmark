import abc
import torch
import pytorch_lightning as pl
from gnn_benchmark.common.utils import num_graphs

class BaseTask(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(self, model, data_path, build_optimizer, device):
        super().__init__()
        self.dataset = self.load_dataset(data_path)
        self.model = model
        self.build_optimizer = build_optimizer
        self.device = device

    def eval_on(self, loader, trainer):
        return trainer.evaluate(self, [loader], max_batches=len(loader))

    def forward(self, *args, **kwargs):
        self.model(*args, **kwargs)

    def configure_optimizers(self):
        return self.build_optimizer(self.model.parameters())


class GraphClassificationTask(BaseTask, metaclass=abc.ABCMeta):
    def training_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = {
            'loss': loss,
            'training_acc': metrics["acc"],
            'n_corrects': metrics["n_corrects"],
            'n_graphs': metrics["n_graphs"]
        }
        return output

    def compute_loss_metrics(self, batch):
        out = self.model.forward(batch)
        loss = self.loss(out, batch.y)
        pred = torch.argmax(out, dim=1)
        acc = torch.mean((pred == batch.y).float())
        pred = torch.argmax(out, dim=1)
        n_corrects = torch.sum((pred == batch.y).int())
        n_graphs = num_graphs(batch)
        metrics = {
            "acc": acc,
            "n_corrects": n_corrects,
            "n_graphs": torch.tensor(n_graphs)
        }
        return loss, metrics

    def loss(self, out, y):
        return torch.nn.functional.cross_entropy(out, y)

    def validation_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = dict({
            'loss': loss,
            'n_corrects': metrics["n_corrects"],
            'n_graphs': metrics["n_graphs"]
        })
        return output

    def validation_end(self, outputs):
        loss_mean = 0
        n_corrects = 0
        n_graphs = 0
        for output in outputs:
            loss_mean += output['loss'].item()
            n_corrects += output['n_corrects']
            n_graphs += output['n_graphs']

        loss_mean /= len(outputs)
        acc = n_corrects.item() / n_graphs.item()

        results = {
            'loss': loss_mean,
            'acc': acc,
        }
        return results


class GraphMultiLabelClassification(BaseTask, metaclass=abc.ABCMeta):
    def loss(self, out, y):
        is_valid = y == y
        return torch.nn.functional.binary_cross_entropy_with_logits(out[is_valid], y.float()[is_valid])

    def training_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = {
            'loss': loss,
            'training_acc': metrics["acc"],
            'n_corrects': metrics["n_corrects"],
            'n_labels': metrics["n_labels"]
        }
        return output

    def compute_loss_metrics(self, batch):
        out = self.model.forward(batch)
        loss = self.loss(out, batch.y)
        pred = (out > 0.5).int()
        acc = torch.mean((pred == batch.y).float())
        n_corrects = torch.sum((pred == batch.y).int())
        n_labels = num_graphs(batch) * batch.y.shape[1]
        metrics = {
            "acc": acc,
            "n_corrects": n_corrects,
            "n_labels": torch.tensor(n_labels)
        }
        return loss, metrics

    def validation_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = dict({
            'loss': loss,
            'n_corrects': metrics["n_corrects"],
            'n_labels': metrics["n_labels"]
        })
        return output

    def validation_end(self, outputs):
        loss_mean = 0
        n_corrects = 0
        n_labels = 0
        for output in outputs:
            loss_mean += output['loss'].item()
            n_corrects += output['n_corrects']
            n_labels += output['n_labels']

        loss_mean /= len(outputs)
        acc = n_corrects.item() / n_labels.item()

        results = {
            'loss': loss_mean,
            'acc': acc,
        }
        return results


class NodeClassificationTask(BaseTask, metaclass=abc.ABCMeta):
    def training_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = {
            'loss': loss,
            'training_acc': metrics["acc"],
            'n_corrects': metrics["n_corrects"],
            'n_nodes': metrics["n_nodes"]
        }
        return output

    def compute_loss_metrics(self, batch):
        out = self.model.forward(batch)
        mask = batch.mask
        out = out[mask]
        y = batch.y[mask]
        loss = self.loss(out, y)
        pred = torch.argmax(out, dim=1)
        acc = torch.mean((pred == y).float())
        pred = torch.argmax(out, dim=1)
        n_corrects = torch.sum((pred == y).int())
        metrics = {
            "acc": acc,
            "n_corrects": n_corrects,
            "n_nodes": torch.sum(mask)
        }
        return loss, metrics

    def loss(self, out, y):
        return torch.nn.functional.cross_entropy(out, y)

    def validation_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = dict({
            'loss': loss,
            'n_corrects': metrics["n_corrects"],
            'n_nodes': metrics["n_nodes"]
        })
        return output

    def validation_end(self, outputs):
        loss_mean = 0
        n_corrects = 0
        n_nodes = 0
        for output in outputs:
            loss_mean += output['loss'].item()
            n_corrects += output['n_corrects']
            n_nodes += output['n_nodes']

        loss_mean /= len(outputs)
        acc = n_corrects.item() / n_nodes.item()

        results = {
            'loss': loss_mean,
            'acc': acc,
        }
        return results


class NodeMultiLabelClassification(BaseTask, metaclass=abc.ABCMeta):
    def loss(self, out, y):
        is_valid = y == y
        return torch.nn.functional.binary_cross_entropy_with_logits(out[is_valid], y.float()[is_valid])

    def training_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = {
            'loss': loss,
            'training_acc': metrics["acc"],
            'n_corrects': metrics["n_corrects"],
            'n_labels': metrics["n_labels"]
        }
        return output

    def compute_loss_metrics(self, batch):
        out = self.model.forward(batch)
        loss = self.loss(out, batch.y)
        pred = (out > 0.5).int()
        acc = torch.mean((pred == batch.y).float())
        n_corrects = torch.sum((pred == batch.y).int())
        n_labels = num_graphs(batch) * batch.y.shape[1]
        metrics = {
            "acc": acc,
            "n_corrects": n_corrects,
            "n_labels": torch.tensor(n_labels)
        }
        return loss, metrics

    def validation_step(self, batch, batch_nb):
        loss, metrics = self.compute_loss_metrics(batch)
        output = dict({
            'loss': loss,
            'n_corrects': metrics["n_corrects"],
            'n_labels': metrics["n_labels"]
        })
        return output

    def validation_end(self, outputs):
        loss_mean = 0
        n_corrects = 0
        n_labels = 0
        for output in outputs:
            loss_mean += output['loss'].item()
            n_corrects += output['n_corrects']
            n_labels += output['n_labels']

        loss_mean /= len(outputs)
        acc = n_corrects.item() / n_labels.item()

        results = {
            'loss': loss_mean,
            'acc': acc,
        }
        return results
