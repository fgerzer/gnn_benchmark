# Results

This document summarizes a few results of different runs, so that you
do not have to run them yourself. These might be produced by (potentially
significantly) older code than the current master.


## Citation Tasks

### GNN Layer Benchmarks

Each of these refers to the best result of the hyperparameter search.

#### Accuracy

| conv_type   |   CiteseerTask |   CoraTask |   PubmedTask |
|:------------|---------------:|-----------:|-------------:|
| GCNConv     |          0.64  |      0.777 |        0.773 |
| GATConv     |          0.634 |      0.76  |        0.755 |
| GINConv     |          0.521 |      0.671 |        0.748 |
| SAGEConv    |          0.62  |      0.767 |        0.774 |
| MLPConv     |          0.469 |      0.457 |        0.705 |

#### Relative Accuracy

| conv_type   |   CiteseerTask |   CoraTask |   PubmedTask |     mean |
|:------------|---------------:|-----------:|-------------:|---------:|
| GCNConv     |       1        |   1        |     0.998708 | 0.999569 |
| GATConv     |       0.990625 |   0.978121 |     0.975452 | 0.981399 |
| GINConv     |       0.814063 |   0.863578 |     0.966408 | 0.88135  |
| SAGEConv    |       0.96875  |   0.98713  |     1        | 0.985293 |
| MLPConv     |       0.732812 |   0.58816  |     0.910853 | 0.743942 |

#### Ranking

| conv_type   |   CiteseerTask |   CoraTask |   PubmedTask |   total |
|:------------|---------------:|-----------:|-------------:|--------:|
| GCNConv     |              1 |          1 |            2 | 1.33333 |
| GATConv     |              2 |          3 |            3 | 2.66667 |
| GINConv     |              4 |          4 |            4 | 4       |
| SAGEConv    |              3 |          2 |            1 | 2       |
| MLPConv     |              5 |          5 |            5 | 5       |

#### Runtimes

| run_definition.conv_type   |   CiteseerTask |   CoraTask |   PubmedTask |
|:---------------------------|---------------:|-----------:|-------------:|
| GCNConv                    |        24.6754 |    9.69428 |      49.7344 |
| GATConv                    |        48.3403 |   11.8244  |      27.6369 |
| GINConv                    |        30.1686 |   11.5794  |      40.8985 |
| SAGEConv                   |        25.6321 |    7.67433 |      24.1542 |
| MLPConv                    |        26.5267 |   14.2105  |      26.2814 |

#### GPU Memory Usage (MB)

| conv_type   |   CiteseerTask |   CoraTask |   PubmedTask |
|:------------|---------------:|-----------:|-------------:|
| GCNConv     |            107 |        113 |          183 |
| GATConv     |             99 |        211 |          430 |
| GINConv     |            125 |        123 |          199 |
| SAGEConv    |            148 |        145 |           75 |
| MLPConv     |            127 |        127 |           69 |
