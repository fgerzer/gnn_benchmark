from gnn_benchmark.common.run_db import DBPaths
from gnn_benchmark.common.utils import create_parameter_combinations, create_fold_configs
from gnn_benchmark.common.arg_parsing import create_db_parser, read_param_file
from pathlib import Path
from gnn_benchmark.modules.gnn_block_net import GNNBlockNetGraph
from gnn_benchmark.tasks import tu_tasks
from torch import nn
from gnn_benchmark.modules.gnn_block import GNNBlock, JumpingBlocks
from gnn_benchmark.modules import conv as geom_conv
from gnn_benchmark.modules import global_poolings
from gnn_benchmark.modules.encoders.linear_encoder import LinearEncoder
from gnn_benchmark.common.definitions import Channels
import torch
from gnn_benchmark.common.utils import choose_gpus
from gnn_benchmark.benchmarking.base_benchmark import BaseBenchmark
from gnn_benchmark.benchmarking.analysis import FoldedAnalysis


class BenchmarkGlobalPoolings(BaseBenchmark):
    tasks = {t.__name__: t for t in [
        tu_tasks.MUTAGTask,
        tu_tasks.PROTEINSTask,
        tu_tasks.IMDBBinaryTask,
        tu_tasks.REDDITBinaryTask,
        tu_tasks.COLLABTask,
    ]}

    global_pools = {p.__name__: p for p in [
        global_poolings.GlobalMeanPooling,
        global_poolings.GlobalAddPooling,
        global_poolings.GlobalMaxPooling,
        global_poolings.GlobalAllPooling,
        global_poolings.GlobalAttentionPooling,
        global_poolings.GlobalSortPooling
    ]}

    def build_model(self, feature_channels, global_pool_type, target_channels, n_hiddens, n_layers, p_dropout, batch_norm,
                    jumping_mode, global_pool_kwargs=None):
        encoder = LinearEncoder(in_channels=feature_channels, out_channels=Channels(node=n_hiddens))
        blocks = []
        global_pool_kwargs = global_pool_kwargs or {}
        in_channels = encoder.out_channels
        for _ in range(n_layers):
            conv = geom_conv.GCNConv(in_channels=in_channels, out_channels=Channels(node=n_hiddens))
            block = GNNBlock(conv=conv, p_dropout=p_dropout, batch_norm=batch_norm)
            blocks.append(block)
            in_channels = block.out_channels
        blocks = nn.ModuleList([JumpingBlocks(nn.ModuleList(blocks), jumping_mode=jumping_mode)])
        global_pooling = self.global_pools[global_pool_type](in_channels=blocks[-1].out_channels, **global_pool_kwargs)
        output_layer = nn.Sequential(
            nn.Linear(global_pooling.out_channels.graph, n_hiddens),
            nn.ReLU(),
            nn.Linear(n_hiddens, target_channels.graph)
        )
        model = GNNBlockNetGraph(
            encoder=encoder,
            blocks=blocks,
            global_pooling=global_pooling,
            output_layer=output_layer
        )
        return model

    def exec_single_run(self, task_name, global_pool_type, n_hiddens, n_layers, p_dropout, batch_norm, folds,
                        fold_idx, jumping_mode, global_pool_kwargs=None):
        task_class = self.tasks[task_name]
        batch_size = 128
        device, gpus = choose_gpus()
        num_workers = 0
        max_epochs = 100

        lr = 1e-3
        weight_decay = 0
        lr_decay_step_size = 100
        lr_decay_factor = 0.5

        def configure_optimizer(parameters):
            optimizer = torch.optim.Adam(
                parameters,
                lr=lr,
                weight_decay=weight_decay
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=lr_decay_step_size,
                gamma=lr_decay_factor
            )
            return [optimizer], [lr_scheduler]

        model = self.build_model(
            feature_channels=task_class.feature_channels,
            target_channels=task_class.target_channels,
            n_hiddens=n_hiddens,
            n_layers=n_layers,
            p_dropout=p_dropout,
            batch_norm=batch_norm,
            global_pool_type=global_pool_type,
            jumping_mode=jumping_mode,
            global_pool_kwargs=global_pool_kwargs
        )

        task = task_class(model=model, data_path=self.data_path, build_optimizer=configure_optimizer,
                                  device=device, folds=folds, fold_idx=fold_idx, batch_size=batch_size,
                                  num_workers=num_workers, cleaned=False)
        run_results = self.run_and_eval(task, max_epochs, gpus)
        return run_results

    def create_params(self):
        max_runs = 100
        n_folds = 10

        base_variable_parameters = {
            "n_layers": [1, 2, 3, 4, 5],
            "n_hiddens": [16, 32, 64, 128],
            "batch_norm": [True, False],
            "p_dropout": [0, 0.5],
            "jumping_mode": [None, "cat"]
        }

        sort_pool_parameters = {
            "global_pool_kwargs.k": [10, 50, 100],
        }

        model_params = {
            "GlobalMeanPooling": base_variable_parameters,
            "GlobalAddPooling": base_variable_parameters,
            "GlobalMaxPooling": base_variable_parameters,
            "GlobalAllPooling": base_variable_parameters,
            "GlobalAttentionPooling": base_variable_parameters,
            "GlobalSortPooling": {**base_variable_parameters, **sort_pool_parameters},
        }

        param_configs = []

        for task in self.tasks:
            for global_pool_type in model_params:
                run_definition = {
                    "task_name": task,
                    "global_pool_type": global_pool_type
                }
                run_definitions = create_parameter_combinations(
                    run_definition, model_params[global_pool_type], max_runs=max_runs
                )

                param_configs.extend(create_fold_configs(run_definitions, n_folds=n_folds))
        return param_configs

    def eval_runs(self):
        interesting_col = "run_definition.global_pool_type"
        metric_col = "results.test_metrics.acc"
        analysis = FoldedAnalysis(self.runs_db, metric_col, metric_comp="max")
        analysis.print_default_analysis(interesting_col, metric_col)

def main():
    parser = create_db_parser()
    parser.add_argument("--create_params", action="store_true")
    parser.add_argument("--worker_loop", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--data_path", type=Path)

    args = parser.parse_args()
    db_collection = args.db_collection or read_param_file()["full_exp_name"]
    db_paths = DBPaths(host=args.db_host, database=args.db_database, collection=db_collection)
    benchmark = BenchmarkGlobalPoolings(db_paths, data_path=args.data_path)
    if args.create_params:
        # params = benchmark.create_params()
        benchmark.create_runs()
    if args.worker_loop:
        benchmark.worker_loop()
    if args.eval:
        benchmark.eval_runs()


if __name__ == '__main__':
    main()
