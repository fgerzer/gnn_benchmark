from gnn_benchmark.common.run_db import RunState
import collections
from gnn_benchmark.common.utils import run_entries_to_df, confidence_interval_95
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import lines
import functools
import copy

class Analysis:
    task_col = "run_definition.task_name"
    time_col = "results.duration"
    mem_usage_col = "results.gpu_mem_usage"


    def __init__(self, runs_db, metric_col, metric_comp="max"):
        self.runs_db = runs_db
        self.metric_col = metric_col
        assert metric_comp in ["max", "min"]
        self.metric_comp = metric_comp

    @functools.lru_cache()
    def _read_runs(self):
        n_runs = self.runs_db.n_runs()
        n_finished = self.runs_db.n_runs(RunState.finished)
        if n_runs > n_finished:
            print(f"\n\nNot all runs finished! "
                  f"Currently, {n_finished}/{n_runs} are finished ({(100 * n_finished) // n_runs}%)\n\n")
        runs_df = run_entries_to_df(self.runs_db.find_finished(), replace_none="None")
        return runs_df

    def _best_run_indices(self, runs_df, compare_col):
        """Computes the indices of the best runs for the interesting parameter"""
        best_indices = []
        model_names = runs_df[compare_col].unique()
        op = "idxmax" if self.metric_comp == "max" else "idxmin"
        for m in model_names:
            best_indices.append(
                getattr(runs_df[runs_df[compare_col] == m][self.metric_col], op)()
            )
        return best_indices

    def _get_param_of_best_run(self, compare_col, param):
        cmp = self.best_runs_df(compare_col)
        cmp = cmp.reset_index(level=[1])
        tasks = cmp.index.unique()

        evaluation_results = {}
        for d in tasks:
            best_run_rows = cmp[cmp.index == d]

            best_run_rows = best_run_rows.set_index(
                compare_col, drop=True
            )
            evaluation_results[d] = best_run_rows[param]

        best_summarized = pd.concat(evaluation_results, axis=1)
        return best_summarized


    def best_results_df(self, compare_col):
        """Gives a high-level overview dataframe containing the performances of the compare_col x the tasks"""
        return self._get_param_of_best_run(compare_col, self.metric_col)

    def runtimes_df(self, compare_col):
        """Gives a high-level overview dataframe containing the runtimes of the best compare_col x the tasks"""
        return self._get_param_of_best_run(compare_col, self.time_col)

    def mem_usage_df(self, compare_col):
        """Gives a high-level overview dataframe containing the memory usage of the best compare_col x the tasks"""
        return self._get_param_of_best_run(compare_col, self.mem_usage_col) // 1024 // 1024


    def best_runs_df(self, compare_col):
        """Returns, for every task/compare_col combination, the best run and its results"""
        runs_df = self._read_runs()
        tasks = runs_df[self.task_col].unique()
        best_hparams = {}
        for d in tasks:
            best_run_idxes = self._best_run_indices(runs_df[runs_df[self.task_col] == d], compare_col)
            best_run_rows = runs_df.loc[best_run_idxes]

            best_run_rows = best_run_rows.set_index(
                compare_col, drop=True
            )
            best_hparams[d] = best_run_rows
        best_hparams = pd.concat(best_hparams, axis=0)
        return best_hparams

    def human_readable(self, df):
        def edit_string(s):
            if s is None:
                return s
            s = s.replace("run_definition.", "")
            s = s.replace("results.", "")
            s = s.replace("_metrics.", ".")
            return s

        df = copy.deepcopy(df)
        columns = df.columns
        if isinstance(columns, pd.MultiIndex):
            for i, level_names in enumerate(columns.levels):
                new_names = [edit_string(n) for n in level_names]
                columns = columns.set_levels(new_names, level=i)
        else:
            columns = columns.to_list()
            for i, c in enumerate(columns):
                c = edit_string(c)
                columns[i] = c
        df.columns = columns
        df.index.name = edit_string(df.index.name)
        return df

    def ranking_df(self, compare_col):
        best_summarized = self.best_results_df(compare_col)
        finished_cols = best_summarized.columns[(~pd.isna(best_summarized).any(axis=0)).values.nonzero()]
        ranking = best_summarized[finished_cols].rank(ascending=self.metric_comp == "min")
        mean_ranking = ranking.mean(axis=1)
        ranking["total"] = mean_ranking
        return ranking

    def relative_performance(self, compare_col):
        best_summarized = self.best_results_df(compare_col)
        if self.metric_comp == "max":
            max_performances = best_summarized.max(axis=0)
        else:
            max_performances = best_summarized.min(axis=0)

        relative_performances = best_summarized / max_performances
        mean_relative_performance = relative_performances.mean(axis=1)
        relative_performances["mean"] = mean_relative_performance
        return relative_performances

    def _plot_overfitting_task(self, df, compare_col, metric_x, metric_y, ax=None, jitter_x=0., jitter_y=0.,
                               same_scale=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        x = np.array(df[metric_x])
        x = x + np.random.normal(0, jitter_x, x.shape)
        y = np.array(df[metric_y])
        y = y + np.random.normal(0, jitter_y, y.shape)
        hue = df[compare_col]
        ax = sns.scatterplot(x=x, y=y, hue=hue,
                             alpha=0.5, ax=ax)
        ax.set_xlabel(metric_x)
        ax.set_ylabel(metric_y)
        if same_scale:
            lims = list(zip(ax.get_xlim(), ax.get_ylim()))
            newlims = min(lims[0]), max(lims[1])
            diagonal = lines.Line2D(newlims, newlims, c=(0, 0, 0, 0.1))
            ax.add_line(diagonal)
            ax.set_xlim(newlims)
            ax.set_ylim(newlims)
        # Setting equal size
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        asp = abs((xmax - xmin) / (ymax - ymin))
        ax.set_aspect(asp)
        return ax

    def overfitting_fig(self, compare_col, metric_x, metric_y, jitter_x=0., jitter_y=0., same_scale=False):
        df = self._read_runs()
        tasks = df[self.task_col].unique()
        ntasks = len(tasks)
        if ntasks <= 3:
            ncols = ntasks
            nrows = 1
        elif ntasks <= 6:
            ncols = 3
            nrows = 2
        else:
            nrows = int(np.ceil((len(tasks) / 1.5)**0.5))
            ncols = int(np.ceil(nrows * 1.5)) - 1
        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
        for ax, t in zip(axes.flatten(), tasks):
            self._plot_overfitting_task(
                df[df[self.task_col] == t], compare_col, metric_x, metric_y, ax=ax, jitter_x=jitter_x,
                jitter_y=jitter_y, same_scale=same_scale
            )
            ax.set_title(t)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')
        [ax.get_legend().remove() for ax in axes.flatten() if ax.get_legend() is not None]
        return fig

    def print_default_analysis(self, interesting_col, metric_col):
        best_results_df = self.human_readable(self.best_results_df(interesting_col))
        best_runs_df = self.human_readable(self.best_runs_df(interesting_col))
        ranking = self.human_readable(self.ranking_df(interesting_col))
        overfitting_fig = self.overfitting_fig(
            compare_col=interesting_col,
            metric_x=metric_col.replace("test_metrics", "train_metrics"),
            metric_y=metric_col,
            same_scale=True
        )
        relative = self.human_readable(self.relative_performance(interesting_col))
        runtimes = self.runtimes_df(interesting_col)

        mem_usage = self.human_readable(self.mem_usage_df(interesting_col))
        with pd.option_context("display.width", 0):
            print("run summary")
            print(best_results_df)
            print("\n\nconfigs of the best runs")
            print(best_runs_df)
            print("\n\nranking")
            print(ranking)
            print("\n\nrelative performance")
            print(relative)
            print("\n\nruntimes (s)")
            print(runtimes)
            print("\n\nGPU mem_usage (MB)")
            print(mem_usage)
        plt.show()


class FoldedAnalysis(Analysis):
    # TODO: Print out confidences in the overview evaluation
    fold_idx_col = "run_definition.fold_idx"

    def _unique_hparams(self, df):
        run_def_cols = [
            c for c in df.columns if c.startswith("run_definition.")
                                     and c != self.fold_idx_col
        ]
        filtered_hparam_columns = []
        for h in run_def_cols:
            if isinstance(df[h].iloc[0], collections.abc.Hashable):
                if len(df[h].unique()) > 1:
                    filtered_hparam_columns.append(h)
            else:
                if len(df[h].transform(tuple).unique()) > 1:
                    filtered_hparam_columns.append(h)
        return filtered_hparam_columns

    def _create_hparam_hash(self, df, to_keep=None):
        # creates a fake "hyperparameter hash" that uniquely defines hparams. This allows us to find all related folds
        # we look for all columns in which there are runs that differ, to later build a string representation (for each run)
        # of which hyperparameter they differ in.
        to_keep = to_keep or set()
        filtered_hparam_columns = self._unique_hparams(df)
        filtered_hparam_columns = list(set(filtered_hparam_columns).union(set(to_keep)))
        return ["|".join(v) for v in df[filtered_hparam_columns].astype(str).values]

    def _statistics_by_fold(self, runs_df, to_keep=None):
        to_keep = to_keep or []
        metrics = [c for c in runs_df.columns if c.startswith("results.")]
        run_parameters = [c for c in runs_df.columns if c.startswith("run_definition.")]

        def create_new_run(cur_run, agg_vals, extracted_runs):
            concats = pd.concat(agg_vals, axis=1).T
            mean_dict = concats.mean().to_dict()
            std_dict = concats.agg(confidence_interval_95).to_dict()
            conf_dict = {k + ".conf": v for k, v in std_dict.items() if np.isfinite(v)}
            extracted_runs.append({**cur_run, **mean_dict, **conf_dict})

        extracted_runs = []
        runs_df["hparam_config"] = self._create_hparam_hash(
            runs_df, to_keep=to_keep
        )
        runs_df = runs_df.sort_values(by="hparam_config")

        cur_run = None
        agg_vals = []
        cur_hparam_config = None

        for (_, row), (_, metrics_row) in zip(runs_df.iterrows(), runs_df[metrics].iterrows()):
            if cur_hparam_config is None or cur_hparam_config != row["hparam_config"]:
                if cur_hparam_config is not None:
                    create_new_run(cur_run, agg_vals, extracted_runs)
                cur_run = row[run_parameters].to_dict()
                cur_hparam_config = row["hparam_config"]
                agg_vals = []
            agg_vals.append(metrics_row)
        create_new_run(cur_run, agg_vals, extracted_runs)

        return pd.DataFrame(extracted_runs)

    @functools.lru_cache()
    def best_runs_df(self, compare_col):
        """Returns, for every task/compare_col combination, the best run and its results"""
        runs_df = self._read_runs()
        runs_df = self._statistics_by_fold(runs_df, to_keep=[compare_col])
        tasks = runs_df[self.task_col].unique()
        best_hparams = {}
        for d in tasks:
            best_run_idxes = self._best_run_indices(runs_df[runs_df[self.task_col] == d], compare_col)
            best_run_rows = runs_df.loc[best_run_idxes]

            best_run_rows = best_run_rows.set_index(
                compare_col, drop=True
            )
            best_hparams[d] = best_run_rows
        best_hparams = pd.concat(best_hparams, axis=0)
        return best_hparams

    def best_results_df(self, compare_col, return_conf=False):
        """Gives a high-level overview dataframe containing the performances of the compare_col x the tasks"""
        if return_conf:
            return self._get_param_of_best_run(compare_col, [self.metric_col, self.metric_col + ".conf"])
        else:
            return self._get_param_of_best_run(compare_col, self.metric_col)

    def print_default_analysis(self, interesting_col, metric_col):
        best_results_df = self.human_readable(self.best_results_df(interesting_col, return_conf=True))
        best_runs_df = self.human_readable(self.best_runs_df(interesting_col))
        ranking = self.human_readable(self.ranking_df(interesting_col))
        overfitting_fig = self.overfitting_fig(
            compare_col=interesting_col,
            metric_x=metric_col.replace("test_metrics", "train_metrics"),
            metric_y=metric_col,
            same_scale=True
        )
        relative = self.human_readable(self.relative_performance(interesting_col))
        runtimes = self.runtimes_df(interesting_col)
        mem_usage = self.human_readable(self.mem_usage_df(interesting_col))
        with pd.option_context("display.width", 0):
            print("run summary")
            print(best_results_df)
            print("\n\nconfigs of the best runs")
            print(best_runs_df)
            print("\n\nranking")
            print(ranking)
            print("\n\nrelative performance")
            print(relative)
            print("\n\nruntimes (s)")
            print(runtimes)
            print("\n\nGPU mem_usage (MB)")
            print(mem_usage)
        plt.show()
