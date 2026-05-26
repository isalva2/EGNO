# Adapting EGNO for Traffic PDEs on Road Networks

A short note on what to keep, what to drop, and what to replace when reusing EGNO's scaffolding as a learned neural operator for LWR / ARZ-type traffic flow on a graph.

## Goal

Treat the road network as a static graph of cells (links or sub-links) and learn an operator $\mathcal{S}_{\Delta t}: u^t \mapsto u^{t+\Delta t}$ for the traffic state
- $u = \rho$ (LWR — scalar density), or
- $u = (\rho, q)$ / $(\rho, w)$ (ARZ — density + a momentum-like variable),
such that the learned kernel reduces, in some limit, to a Godunov-type scheme for the underlying PDE on the network.

## What does *not* transfer from EGNO

| Component | Why it's wrong for traffic |
|---|---|
| $O(3)$ equivariant message passing ([InvariantScalarNet](model/basic.py)) | LWR/ARZ have no 3D rotation symmetry. The $Z^TZ$ inner products encode the wrong invariance group. |
| Position channel `x` and velocity channel `v` | Network geometry is static; node "position" never updates. `v = dx/dt` has no traffic analogue. |
| `loc_mean` translation trick | Translation equivariance is not the right gauge — mass conservation is. |
| Time-Fourier kernel (`TimeConv`) | Spectral parametrization handles smooth, periodic, dispersive dynamics. LWR/ARZ are nonlinear hyperbolic and form shocks; Fourier truncation triggers Gibbs ringing exactly in the regime of interest (stop-and-go waves, jamitons). |

## What *does* transfer

- **Embed-once-then-time-convolve pattern.** Single graph embedding lifted to $T$ timesteps, time mixing at each layer.
- **Multi-frame one-shot output.** Predicting $T$ future cell states per forward pass — useful for short-horizon traffic forecasting and consistent with operator-learning framings.
- **Layered spatial / temporal alternation.** The interleaving in `EGNO.forward` of time-conv + spatial-conv blocks is a reasonable structure for a spacetime neural operator; just replace the building blocks.

## Replacement design sketch

### Node state
- LWR: `u_i ∈ ℝ` (density on cell $i$, normalized by jam density).
- ARZ: `u_i ∈ ℝ²` — either $(\rho, \rho v)$ or $(\rho, w)$ with $w = v + p(\rho)$.

### Edge messages = numerical fluxes
Replace `InvariantScalarNet` with a flux network $F_\theta(u_i, u_j, e_{ij}) \to \mathbb{R}^{d_u}$, where $e_{ij}$ encodes link-specific parameters (free-flow speed $v_f$, capacity, length, jam density). Constrain it to a Riemann-solver shape:

- For LWR (Godunov): $F = \min(S(u_i), R(u_j))$ where $S$ (sending), $R$ (receiving) are learned monotone functions of upstream/downstream density. Recovers the Cell Transmission Model when $S, R$ match the analytical fundamental diagram triangle.
- For ARZ: a two-component Roe- or HLL-style flux with a learned eigenstructure.

### Node update = conservative aggregation
$$ u_i^{t+\Delta t} = u_i^t - \frac{\Delta t}{L_i} \sum_{j \in \partial i} \sigma_{ij}\, F_\theta(u_i, u_j, e_{ij}) $$
with $\sigma_{ij} = +1$ for outgoing edges, $-1$ for incoming. **Mass conservation is built in** because every $F_{ij}$ appears with opposite signs on its two endpoints.

### Junctions
Treat multi-edge nodes (intersections) as a separate edge kernel $F^{\text{junc}}_\phi$ with priority / merging logic baked in (e.g. Daganzo's incremental transfer principle). This is where the graph structure earns its keep over a 1D PDE solver.

### Temporal kernel
Drop `TimeConv`'s Fourier basis. Options:
- A monotone time stepper (learned IMEX integrator with TVD/monotonicity constraints).
- A learned flux limiter parameterized over a small set of stencil weights.
- For pure operator-learning ambitions, keep an FNO-style time kernel but add a slope-limiter postprocessor to suppress Gibbs near shocks.

## Suggested ablations / theoretical targets

1. **Consistency limit.** Show that as `hidden_nf → ∞` and training data → ∞, the learned $F_\theta$ converges to the analytical Godunov flux for a chosen fundamental diagram (or characterize the bias).
2. **Conservation by construction vs. as a soft penalty.** Compare hard structural conservation (above) to an unconstrained GNN with a conservation loss term.
3. **Shock fidelity.** Stress-test on classical LWR Riemann problems on a single link — does the learned kernel preserve shock speed (Rankine-Hugoniot) and entropy conditions?
4. **Network effects.** On a small ring or diamond network with known analytical / CTM solutions, quantify how much the junction kernel adds over treating links independently.
5. **Hyperbolicity preservation for ARZ.** Verify the learned 2×2 flux Jacobian stays hyperbolic; otherwise the model can develop unphysical instabilities.

## References to anchor against

- Brandstetter, Worrall, Welling — *Message Passing Neural PDE Solvers* (2022).
- Li, Kovachki, Azizzadenesheli, et al. — *Graph Neural Operator* / *Fourier Neural Operator* line.
- Daganzo — *The Cell Transmission Model* (1994, 1995): the discrete LWR scheme your spatial kernel should asymptote to.
- Aw & Rascle (2000); Zhang (2002): the ARZ system.
- Garavello & Piccoli — *Traffic Flow on Networks*: junction Riemann solvers, which become your junction edge kernel.
- Cuomo et al. — survey of *Scientific Machine Learning* / PINOs, for the operator-learning framing.

## Bottom line

What's worth borrowing from EGNO is the **spacetime layered operator with one-shot multi-frame output**, not its specific kernels. Swap equivariant message passing for **flux-form conservative message passing**, swap time-Fourier for a **shock-aware temporal kernel**, and the natural theoretical contribution is an *equivariant-to-the-right-group* neural operator on networks whose learned kernel reduces to a Godunov/Riemann-solver scheme for LWR/ARZ in a controllable limit.
