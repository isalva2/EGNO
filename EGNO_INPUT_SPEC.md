# EGNO Input Specification

How to convert arbitrary graph data into a valid input for the EGNO model.

EGNO predicts future positions and velocities of a set of points connected by a graph, in 3D space. One forward pass takes a single snapshot and returns `num_timesteps` future snapshots.

## Forward signature

From [model/egno.py:28](model/egno.py#L28):

```python
loc_pred, vel_pred, h_out = model(x, h, edge_index, edge_fea, v=v, loc_mean=loc_mean)
```

## Per-sample data you must provide

For a single graph with `N` nodes and `E` directed edges:

| Tensor | Shape | Dtype | Meaning | Constraints |
|---|---|---|---|---|
| `x` | `[N, 3]` | float | 3D position of each node | world-frame coordinates |
| `v` | `[N, 3]` | float | 3D velocity of each node | same frame as `x`; can be zeros if unknown |
| `h` | `[N, in_node_nf]` | float | Scalar (O(3)-invariant) node features | width must equal model's `in_node_nf` |
| `edge_index` | `[2, E]` | long | Directed edges as `[rows, cols]` | indices in `[0, N)`; include both `(i,j)` and `(j,i)` if you want symmetric msgs |
| `edge_fea` | `[E, in_edge_nf - 1]` | float | Scalar edge features (types, weights, …) | the model appends `‖x[i]-x[j]‖²` itself, so reserve **one slot** for that and pass `in_edge_nf - 1` columns; final `in_edge_nf` must match init |
| `loc_mean` | `[N, 3]` | float | Per-graph centroid of `x`, broadcast to every node | `x.mean(dim=0, keepdim=True).expand(N, 3)` |

## What counts as valid features

- **`h`**: anything that doesn't change under rotation/reflection of the coordinate frame — speeds, charges, scalar atom types, time embeddings you've added externally, etc. Don't put raw `xyz` here; it would break equivariance.
- **`edge_fea`**: same restriction — distances, bond types, scalar weights. The model itself adds `‖x_i - x_j‖²` as the last column inside the training scripts (see [main_mocap_no.py:214-215](main_mocap_no.py#L214-L215)), so plan for that slot.

## Batching multiple graphs

EGNO has no `batch` argument; instead you concatenate samples into one large disconnected graph:

```python
# B samples, each with N_i nodes and E_i edges
x         = torch.cat([x_i        for x_i in xs],        dim=0)           # [ΣN_i, 3]
v         = torch.cat([v_i        for v_i in vs],        dim=0)           # [ΣN_i, 3]
h         = torch.cat([h_i        for h_i in hs],        dim=0)           # [ΣN_i, in_node_nf]
loc_mean  = torch.cat([loc_mean_i for loc_mean_i in lms],dim=0)           # [ΣN_i, 3]

# offset each sample's edge indices so they point into the concatenated x
offset = 0
ei_rows, ei_cols, efs = [], [], []
for i, ei_i in enumerate(edge_indices):
    ei_rows.append(ei_i[0] + offset)
    ei_cols.append(ei_i[1] + offset)
    efs.append(edge_feas[i])
    offset += xs[i].shape[0]
edge_index = [torch.cat(ei_rows), torch.cat(ei_cols)]                     # two [ΣE_i] tensors
edge_fea   = torch.cat(efs, dim=0)                                        # [ΣE_i, in_edge_nf - 1]
```

If all samples share the same `N` (the common case — fixed skeleton, fixed n-body count), the offset becomes `i * N` and you can vectorize:

```python
offset = (torch.arange(B) * N).unsqueeze(-1).unsqueeze(-1)   # [B, 1, 1]
edges_stacked = torch.stack(edge_indices)                    # [B, 2, E]
edge_index = (edges_stacked + offset).permute(1, 0, 2).reshape(2, -1)  # [2, B*E]
```

## Init-time constraints (`EGNO(...)`)

The model has to know feature widths up front, and they must match your data exactly:

| Init arg | Must equal |
|---|---|
| `in_node_nf` | `h.shape[1]` |
| `in_edge_nf` | `edge_fea.shape[1]` **after** the model adds `‖Δx‖²` (i.e. your provided width + 1) |
| `num_timesteps` | how many future frames you want predicted in one shot |
| `hidden_nf`, `n_layers`, `num_modes`, `time_emb_dim`, `flat` | architectural; only constrained when loading a checkpoint |

## Output

```python
loc_pred.shape == [num_timesteps * ΣN_i, 3]
vel_pred.shape == [num_timesteps * ΣN_i, 3]
```

Layout is `[t=0 batch concat, t=1 batch concat, …]`. To recover `[B, T, N, 3]` when all samples share `N`:

```python
loc_pred.view(num_timesteps, B, N, 3).permute(1, 0, 2, 3)
```

## Equivariance contract (what you get for free, what you must respect)

- **Translation**: rotate/translate `x` (and update `loc_mean` accordingly) and the predicted `loc_pred` transforms the same way. ⚠ if you forget to update `loc_mean`, you break this.
- **Rotation/reflection**: rotate `x` and `v` by the same `R ∈ O(3)`, and predictions rotate consistently. Scalar features (`h`, `edge_fea`) must be invariants — don't sneak coordinates in.
- **Permutation**: relabel nodes (permuting `x`, `v`, `h`, and updating `edge_index` accordingly) — predictions permute the same way.

## Minimal checklist before calling `model(...)`

1. `x.shape[1] == v.shape[1] == 3` ✓
2. `h.shape[1] == model.embedding.in_features - time_emb_dim` ✓ (the model adds the time embedding internally; see [model/egno.py:11](model/egno.py#L11))
3. `edge_fea.shape[1] + 1 == in_edge_nf` ✓ (you provide all but the distance column)
4. `max(edge_index[0]).item() < x.shape[0]` and same for `edge_index[1]` ✓
5. `loc_mean` is per-graph mean broadcast to every node, not a single row ✓
6. Everything is on the same `device` as the model ✓
