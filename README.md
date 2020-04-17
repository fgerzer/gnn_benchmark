# Graph Neural Networks Benchmarks

> **This is a research repository and a work-in-progress.**

This is a repository bundling several benchmarks (in particularly with
the goal of integrating the [OGB](https://ogb.stanford.edu/) benchmarks)
into an easy-to-use framework.

My primary use for this is - obviously - my own research; however, I'd
be happy if people got some use out of it.

## Requirements

### Singularity

The simplest way of using this is to install and use the
[singularity scientific container platform](https://sylabs.io/singularity/).
For this, follow the [instructions](https://sylabs.io/guides/3.5/admin-guide/installation.html)
to install singularity, then build the container using

```setup
sudo singularity build gnn_benchmark.sif gnn_benchmark.singularity
```
which will download and install all required files into the
`gnn_benchmark.sif` container.

> **_WARNING: CUDA VERSIONS_** Depending on your own setup, you might
> need to change the `TORCH_CUDA_ARCH_LIST` variable in the script. My
> own GPUs have compute capability of 5.0 (laptop) and 6.1 (server);
> including both allows me to use the same image for both. If you
> have different cuda versions, you have to change this. Similarly,
> the image uses CUDA 10.1, so you need a fairly recent driver for your
> host.

Additionally, you will need an installation of mongodb - either a
local one, or you can use the one the image comes with (more details
on that later).

### Local Installation

Local installation is handled in the `requirements.txt` - to install
these, use

```setup
pip install -r requirements.txt
```
However, that does not install everything! (You may wonder why - the
reason is that pytorch-geometric apparently needs to be compiled on
singularity;
[more details](https://github.com/rusty1s/pytorch_geometric/issues/923)).

The canonical instructions are found in the `gnn_benchmark.singularity`
file; I've included them here for your convenience but they may be
outdated - the others are always fresh, since they are used to build
 the image I use.

```
export CUDA=cu101
pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html

git clone https://github.com/rusty1s/pytorch_geometric.git && \
    cd ./pytorch_geometric && \
    git checkout 1.4.3 && \
    python3 ./setup.py install && \
    cd ..
```

## General Workflow

The workflow used here is simple - when starting a new experiment, a
mongodb collection is initialized, containing all prospective run
parameters. Afterwards, an arbitrary number of workers can be run,
pulling their work from the database and writing back results once
finished. This should work for large numbers of workers, but I have not
yet tested it with more than a dozen concurrent workers.

## Training

Each experiment (i.e. question to be answered) has its own run and
evaluation script, found in `gnn_benchmark/experiments/`.

### Starting the Database
First, you need to know where the database runs. If you're using your
own, you probably already know this. If you want to use the one packaged
in the container, navigate to its folder, create the `local_db` folder,
and call

`singularity exec --nv -B ./local_db:/data/db mongodb_test.sif mongod`

This starts the database in non-daemon mode (i.e. if you press `CTRL-C`
it will stop). It will write its own database into `./local_db`, and if
you restart it from the same folder, the database will persist.

If you already have a database running and do not want to interfere, call
it with `mongod --port <portnumber>`.

This database must run throughout the training and evaluation process.

### Training the Model

I'll be using the `tu_graph/benchmark_gnn_layers.py` script as an example.

This script is supposed to answer the question "Which GNN layer works
best on a certain subset of TUD graph datasets". That subset is `MUTAG`,
`PROTEINS`, `IMDBBinary`, `REDDITBinary`, and `COLLAB`. The script can
be started with

```
python gnn_benchmark/experiments/tu_graph/benchmark_gnn_layers.py <command line arguments>.
```

There are several command line arguments you have to fill. Specifically:
- `--db_host`: Where does the database run?
- `--db_database`: Which database should we use?
- `--db_collection`: Which collection should we store the results in?
- `--data_path`: Where can the data be found (or should be downloaded to).

These values depend on your configuration. If you use a local database,
it will probably be something like
`--db_host "mongodb://localhost:27017" --db_database gnn_benchmark --db_collection tud_gnn_layers`;
however, the latter two are completely your choice (and if you chose a
different port, you have to change that as well).

`--data_path` is quite simple; however, be aware that I have not tested
what happens if you concurrently download the same data in different
processes. Download it in one, or proceed at your own peril.

Lastly, there are two command line arguments that you can use for training:
- `--create_params` tries to create the parameters for all runs
- `--worker_loop` tries to iteratively read the parameters and return the results.

> **_NOTE_** There is a rough locking mechanism in place. This mechanism
> ensures that you can start any number of processes that create parameters,
> but only one does. This also means that only one process ever creates
> parameters; if that is killed for some reason, you have to delete the
> collection yourself.

So, a full command might look like this:

```train
python gnn_benchmark/experiments/tu_graph/benchmark_gnn_layers.py --data_path ../data --db_host "mongodb://localhost:27017" --db_database gnn_benchmark --db_collection tud_gnn_layers --create_params --worker_loop`
```

Wrapping this in the singularity container is simple: Add
`singularity exec --nv CMD`. It is possible that there is some
interference from your home folder; in that case, you have to explicitly
bind paths:

```
singularity exec --nv -B <host_datapath>:/experiment/data -B <host_codepath>:/experiment/code; bash -C 'cd /experiment/code; python gnn_benchmark/experiments/tu_graph/benchmark_gnn_layers.py --data_path /experiment/data --db_host "mongodb://localhost:27017" --db_database gnn_benchmark --db_collection tud_gnn_layers --create_params --worker_loop'
```

Once everything is finished, you can evaluate it.

## Evaluation

Evaluation can be conducted with the exact same command as above (though
you don't need the data_path), just using the `--eval` flag. In our
example, that's

```eval
singularity exec --nv -B <host_datapath>:/experiment/data -B <host_codepath>:/experiment/code; bash -C 'cd /experiment/code; python gnn_benchmark/experiments/tu_graph/benchmark_gnn_layers.py --data_path /experiment/data --db_host "mongodb://localhost:27017" --db_database gnn_benchmark --db_collection tud_gnn_layers --eval'
```

## Pre-trained Models

There are no pretrained models, and I am currently not saving any.
Several aspects of the current training of GNN models appears to clash
with common assumptions made for Deep Learning. One of these is the
speed of training. Particularly for semi-supervised node classification,
you can often train whole models far faster than you'd need for one
epoch on an image dataset. Accordingly, it simply does not make sense
for me to store the large numbers of models this produces.

## Results

Several results can be found in [the results markdown file](./results.md)

## Contributing

This is licensed under an MIT License; if you have own experiments to
add, you can simply pull-request. On questions, please raise an issue.

## Acknowledgements
To further abuse an oft-used quote, this work is only possible because
I could stand on the shoulders of other libraries. In particular, I'd
like to thank

- Matthias Fey (@rusty1s), for his work on
[pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) - I have
no idea how much time he spends on making that library so impressive,
but it is definitely worth it.
- The [OGB team](https://github.com/snap-stanford/ogb) for more complex
graph benchmarks than we have used before.
