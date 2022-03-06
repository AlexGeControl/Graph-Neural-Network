# Deep Learning on Graphs, 03 Graph Network Robustness

---

## Environment Setup

The virtual environment used is defined by [this Conda environment file](docker/environment/graph/graph.yaml)

---

## Up & Running

First, run the command below inside Docker:

```bash
jupyter-lab --ip=0.0.0.0 --port=6006 --no-browser --allow-root
```

Then the Jupyter notebook will be available at **localhost:46006**

---

## SimpleAttack 

The Jupyter notebook for this assignment is available [here](assignment-deeprobust.ipynb)

### Implementation

#### Removal, with Connected Component Maintained

```Python
def generate_candidates_removal(self, adj):
    """产生减边的候选边：从当前的所有边中除开那些会产生孤立节点的边。
    """
    # first, build a minimum-spanning tree to identify all critical edges:
    mst = sp.csgraph.minimum_spanning_tree(adj)
    mst = mst.maximum(mst.T)
    
    # extract candidate edges:
    candidates = list(
        zip(*(adj - mst).nonzero())
    )
    
    return candidates
```

#### Addition, using Iterative Extension

```Python
def generate_candidates_addition(self, adj, n_candidates, seed=None):
    """产生可以被加边的候选边（也就是说，现在不是边)。
    """
    # set seed for numpy.random if provided:
    if seed is not None:
        np.random.seed(seed)

    # generate candidates:
    N = len(adj.indptr) - 1
    
    candidates = set()
    while len(candidates) < n_candidates:
        # generate edge proposal:
        proposals = np.random.randint(0, N, [n_candidates, 2])
        
        # keep only edges that are not in the current graph:
        proposals = set(
            list(map(tuple, proposals[adj[proposals[:, 0], proposals[:, 1]].A1 < 1.0]))
        )
        
        # extend candidate set:
        candidates = candidates.union(proposals)
    candidates = list(candidates)
    
    return candidates[:n_candidates]
```

#### Random Selection

```Python
def random_top_flips(self, candidates, n_perturbations, seed=None):
    """从candidates中随机选择n_perturbations个候选边。
    """
    # set seed for numpy.random if provided:
    if seed is not None:
        np.random.seed(seed)
        
    return [
        candidates[i] for i in 
        np.random.permutation(len(candidates))[:n_perturbations]
    ]
```

#### Selection by Edge Degree

```Python
def degree_top_flips(self, adj, candidates, n_perturbations):
    """从candidates中随机选择n_perturbations个degree最大的候选边。
    这里，边的degree我们可以计算为它连接的节点的degree的相加。
    """
    # get node degrees:
    N = len(adj.indptr) - 1
    node_degree = {
        k: v for (k, v) in zip(range(N), np.diff(adj.indptr))
    }
    
    # get candidate edge degree
    edge_degree = [
        node_degree[u] + node_degree[v] for (u, v) in candidates
    ]
    
    return [
        candidates[i] for i in 
        # identify most influential edges through fast selection:
        np.argpartition(edge_degree, -n_perturbations)[-n_perturbations:]
    ]
```

#### Flip

```Python
def flip_candidates(self, adj, candidates):
    """翻转候选边，0变成1，1变成0。

    返回值: sp.csr_matrix, shape [n_nodes, n_nodes]
        翻转后的邻接矩阵。
    """
    # num. of nodes:
    N = len(adj.indptr) - 1
    
    # generate pertubation:
    pertubation = sp.coo_matrix(
        (np.ones(len(candidates)), tuple(zip(*candidates))), 
        shape=(N, N)
    ).tocsr()
    
    # flip selected edges:
    adj_flipped = pertubation - adj
    
    # prune:
    adj_flipped.eliminate_zeros()
    
    # done:
    return adj_flipped
```

### Results:

```bash
# conclusion -- add is more destructive than removal
random add
    Micro F1: 0.7484909456740443
    Macro F1: 0.7179212506057302
random remove
    Micro F1: 0.7952716297786722
    Macro F1: 0.7689443654931297
# conclusion -- add is more destructive than removal
degree add
    Micro F1: 0.772635814889336
    Macro F1: 0.7468532574920008
degree remove
    Micro F1: 0.7882293762575453
    Macro F1: 0.754619675083964
```

---
