# Method specification

## Planning variables and reward

A candidate is an unordered set of N transmitter locations. Continuous search
uses a tensor of shape `[particles, N, 2]` in the unit square. Metric coordinates
are recovered by `xy_m = (xy_norm01 - 0.5) * map_size_m`. Transmitter height is
fixed by the radio profile.

The common search objective combines a scene-conditioned predicted reward with
differentiable penalties for nondeployable positions and insufficient pairwise
distance. Final candidates are mapped to feasible road positions before physical
evaluation. Predicted rewards and physical coverage are stored separately.

## Reward representations

| Model | Set representation | Training signal |
| --- | --- | --- |
| HPEM | Dense scene encoder, location features, symmetric order-1 to order-4 hyperedge potentials, and a global term | Joint-coverage regression, within-family ranking, and auxiliary marginal/joint coverage regression |
| Regressor | Scene-conditioned TX embeddings with permutation-invariant pooling | Joint-coverage regression |
| IN | TX nodes with learned pairwise relation effects and invariant pooling | Joint-coverage regression |
| NRI | Categorical latent pairwise edges with relation-specific effects | Joint-coverage regression and a categorical-edge KL regularizer |

HPEM returns an energy E and a reward -E. Each order averages its valid hyperedge
contributions; a global head also receives the order summaries. Enabling multiple
orders does not require multiple independently trained models. The other three
representations return scalar rewards through the same inference interface.

IN and NRI adapt relational reasoning to static deployment sets. NRI uses
Gumbel-Softmax during training and edge probabilities during evaluation; this
implementation does not train a trajectory-reconstruction dynamics VAE.

The scene representation has 11 channels. A dense encoder supplies local
features at TX locations and pooled scene context. Batches are grouped by scene
and cardinality. HPEM ranking comparisons additionally use family identity.

## Sampling and search

All methods accept a callable mapping candidate TX sets to scalar rewards.

| Method | Implementation |
| --- | --- |
| Diffusion | Reverse variance-exploding SDE with a Monte Carlo estimate of a Gaussian-smoothed reward-induced score |
| WS-G | Annealed sequential Monte Carlo with Gaussian random-walk Metropolis mutation |
| WS-L | Annealed sequential Monte Carlo with Metropolis-adjusted Langevin mutation |
| NUTS | No-U-Turn Hamiltonian sampling in unconstrained coordinates, with a sigmoid transform and its Jacobian |
| BR-SNIS | Iterated sampling-importance resampling with a retained conditioning particle and fresh uniform proposals |
| Greedy | Sequential insertion from a candidate pool, with optional coordinate refinement |
| PSO | Particle swarm search with scheduled inertia and cognitive/social updates |
| GA | Scalar-fitness real-coded genetic search with tournament selection, simulated binary crossover, polynomial mutation and elitism |
| TuRBO | Single trust-region Bayesian optimization with a Matérn-5/2 Gaussian process and posterior-sampling candidate selection |

Diffusion evaluates multiple Gaussian neighbors per particle at each noise level.
Their softmax reward weights combine coordinate gradients into a score estimate.
The model is a reward predictor; no separately trained denoising network is
required. Reflected boundary handling keeps candidates in the normalized domain.

WS-G and WS-L use incremental tempering weights, effective-sample-size-triggered
resampling, and mutation steps. NUTS controls trajectory length through tree
expansion and a U-turn condition. BR-SNIS returns concrete candidate sets rather
than averaging coordinates across sets. Its use here is an ISIR-based planning
adaptation of the bias-reduced estimation framework.

Greedy has no submodular approximation guarantee for the interference-coupled
objective. GA optimizes one scalar reward and is not an implementation of
multi-objective NSGA-III. TuRBO uses one adaptive trust region. These distinctions
define the implemented baselines and their scope.

## Physical evaluation

The radio profile specifies 3.65 GHz, 20 MHz bandwidth, 24 dBm per TX, isotropic
single-element vertically polarized antennas, TX height 10 m, and RX height
1.5 m. The measurement grid has 1 m cells. The configured solver uses a maximum
depth of five, five million samples per TX, line of sight, specular and diffuse
reflection, refraction, and diffraction/edge-diffraction options.

Serving association uses the strongest received signal. Interference includes
the other active transmitters. SS-RSRP uses the configured active-subcarrier
power allocation. Effective throughput applies the configured implementation
efficiency, resource share, and spectral-efficiency cap to a Shannon expression;
it is not measured end-user throughput.

| Criterion | Threshold |
| --- | --- |
| Path loss | at most 106 dB |
| SS-RSRP | at least -110 dBm |
| SINR | at least 5 dB |
| Effective throughput | at least 10 Mbit/s |

Joint coverage is the cell-wise intersection of the four criteria, divided by
the number of road evaluation cells. It is neither the product nor the mean of
the marginal coverage fractions. The path-loss and SS-RSRP criteria are related
through the link budget; they are not statistically independent objectives.

## Evaluation protocol

Record the scene, cardinality, reward weights, sampler configuration, seed,
reward-query count, search time, and ray-tracing time for each run. Report sample
standard deviations over repeated seeds. Method-specific budgets are not
automatically equalized by matching population sizes or iteration counts.

For adaptation, target evaluation deployments are reserved before training.
Few-shot training uses 30% of the adaptation pool; full adaptation continues from
that state using the full adaptation pool. Evaluation labels remain excluded.
These are within-target adaptation experiments, distinct from base-model
evaluation on geographically unseen scenes.
