# SDE Data Generation Scripts

This repository contains data generation scripts for the numerical examples used in several stochastic flow map learning papers. The scripts generate training, testing, conditional, and original trajectory data for SDE, multiscale SDE, SPDE, and SSA examples.

Most Python scripts save `.mat` files to the parent directory of this repository using names derived from the script name, for example `Ex3OU_train.mat` and `Ex3OU_test.mat`. The Matlab SSA scripts currently contain several commented generation blocks; choose the block you need and adjust the output path before running.

## References

The scripts cover examples from the following papers:

- [1] Yuan Chen and Dongbin Xiu, `Learning stochastic dynamical system via flow map operator`, 2024.
  https://iamyuanchen.xyz/pdf/2024ChenXiu.pdf
- [2] Zhongshu Xu, Yuan Chen, Qifan Chen, and Dongbin Xiu, `Modeling Unknown Stochastic Dynamical System via Autoencoder`, 2024.
  https://iamyuanchen.xyz/pdf/2024XuChenChenXiu.pdf
- [3] Yanfang Liu, Yuan Chen, Dongbin Xiu, and Guannan Zhang, `A Training-Free Conditional Diffusion Model for Learning Stochastic Dynamical Systems`, 2025.
  https://iamyuanchen.xyz/pdf/2025LiuChenXiuZhang.pdf
- [4] Yuan Chen and Dongbin Xiu, `Modeling Unknown Stochastic Dynamical System Subject to External Excitation`, 2026.
  https://iamyuanchen.xyz/pdf/2026ChenXiu.pdf
- [5] Yuan Chen and Dongbin Xiu, `Data-Driven Effective Modeling of Multiscale Stochastic Dynamical Systems`, 2024.
  https://iamyuanchen.xyz/pdf/2024ChenXiu_b.pdf
- [6] Yuan Chen, Weize Mao, and Dongbin Xiu, `Data-Driven Effective Modeling of Stochastic Chemical Reaction Networks`, to be published soon.

## Requirements

Python scripts require:

- Python 3
- `numpy`
- `scipy`

Matlab scripts require Matlab and the SSA helper functions used by the scripts, including functions such as `examples`, `GenDatawithGillespieSSA`, and `SSA_path_rs`, available on the Matlab path. The dependencies required by `Ex42Vilar2002R.m` and `Ex41SSACIRC73s.m` are not included because they rely on code written by other authors. Please contact `chen.11050@osu.edu` or `yuan_chen1@brown.edu` if you need those files.

## Repository Structure

The repository is organized as follows:

| Path | Contents |
| --- | --- |
| `*.py` | Python data-generation scripts for SDE, multiscale SDE, SPDE, and Python-based SSA examples. |
| `SSA/` | Matlab SSA data-generation scripts. These scripts may require external Matlab helper code that is not included in this repository. |
| `README.md` | Overview of references, scripts, running instructions, and output data format. |

Generated `.mat` data files are not tracked in this repository by default. Most Python scripts write output files to the parent directory of the repository; adjust the save path in each script if you want a different destination.

## Scripts

### Python

Core SDE examples shared by the GAN, autoencoder, and conditional diffusion sFML papers:

| Script | Example | Paper |
| --- | --- | --- |
| `Ex3OU.py` | Ornstein-Uhlenbeck process | [1], [2], [3] |
| `Ex1GeoBrownian.py` | Geometric Brownian motion | [1], [2], [3] |
| `Ex4ExpDiff.py` | SDE with nonlinear/exponential diffusion | [1], [2], [3] |
| `Ex5Trig.py` | Trigonometric drift and diffusion | [1], [2], [3] |
| `Ex8DoubleWell.py` | SDE with double-well potential | [1], [2], [3] |
| `Ex9Expdis.py` | SDE with exponentially distributed noise | [1], [2], [3] |
| `Ex6ExpOU.py` | SDE with lognormally distributed noise | [1], [2], [3] |
| `Ex7MdOU.py` | Multi-dimensional OU process; can be used for 2D or 5D OU | [1], [2], [3] |
| `Ex10SO.py` | Stochastic oscillator | [1], [3] |

Nonautonomous and controlled examples from the 2026 nonautonomous/control paper:

| Script | Example | Paper |
| --- | --- | --- |
| `Ex12DisturbOU.py` | OU process with drift control | [4] |
| `Ex19BiStochsticOU.py` | OU process with both drift and diffusion control | [4] |
| `Ex16Multiscale.py` | Nonlinear SDE with control | [4] |
| `Ex17PredPrey.py` | Stochastic predator-prey / Lotka-Volterra model with excitation | [4] |
| `Ex15StochasticRes.py` | Stochastic resonance / double-well with excitation | [4] |
| `Ex43SSAmRNAwDynk.py` | Gene expression SSA model with time-dependent reaction rate | [4] |
| `SPDEEx3SHeatEquModal_wSource_Spectral.py` | Stochastic heat equation with source, in spectral/modal form | [4] |

Multiscale examples from the multiscale stochastic flow map learning paper:

| Script | Example | Paper |
| --- | --- | --- |
| `Ex28MultiScaleSkewProduct.py` | Skew product SDE | [5] |
| `Ex33MultiScaleExp.py` | Exponential mean OU / multiscale exponential example | [5] |
| `Ex38MultiscaleTriad.py` | Triad system | [5] |
| `Ex34MultiScaleDuan3D.py` | 3D nonlinear SDE | [5] |
| `Ex36MultiscaleNonlinOclator.py` | Multiscale stochastic oscillator | [5] |

SSA examples implemented in Python:

| Script | Example | Paper |
| --- | --- | --- |
| `Ex22SSATransfer.py` | Transfer process | [6] |
| `Ex23SSALV.py` | SSA Lotka-Volterra model | [6] |
| `Ex25SSABrusselator.py` | Brusselator | [6] |
| `Ex27SSAautocatalytic.py` | Autocatalysis | [6] |
| `Ex26SSAOregonator.py` | Oregonator | [6] |
| `Ex45SSASchlogl.py` | Schlogl model | [6] |

Additional SPDE scripts retained in the repository:

| Script | Purpose | Paper |
| --- | --- | --- |
| `SPDEEx1SHeatEqu.py` | Stochastic heat equation data generation | Additional SPDE script |
| `SPDEEx1SHeatEquModal.py` | Modal form of the stochastic heat equation | Additional SPDE script |
| `SPDEEx1SHeatEquModal_Spectral.py` | Spectral/modal stochastic heat equation | Additional SPDE script |
| `SPDEEx2SAdvDiffModal_Spectral.py` | Spectral/modal stochastic advection-diffusion example | Additional SPDE script |

Run a Python script directly from this folder:

```bash
python Ex3OU.py
```

Most Python scripts follow this pattern:

- create training data;
- create testing data;
- save the output as `.mat` files in the parent folder.

For example:

```bash
python Ex3OU.py
```

generates:

```text
../Ex3OU_train.mat
../Ex3OU_test.mat
```

Examples with external controls or parameters usually also save a `para` array:

```bash
python Ex12DisturbOU.py
python Ex19BiStochsticOU.py
python Ex17PredPrey.py
```

SSA Python scripts may additionally contain commented blocks for conditional distributions, original SSA paths, or stopping-time data. Uncomment the block you need before running.

### Matlab

SSA examples implemented in Matlab:

| Script | Example | Paper |
| --- | --- | --- |
| `SSA/Ex42Vilar2002R.m` | Vilar 2002 genetic oscillator model | [6] |
| `SSA/Ex41SSACIRC73s.m` | Mammalian circadian clock model | [6] |

The two Matlab files, `Ex42Vilar2002R.m` and `Ex41SSACIRC73s.m`, depend on code written by other authors. We do not have permission to share those dependencies in this repository. If you need to run these examples, please contact the author at `chen.11050@osu.edu` or `yuan_chen1@brown.edu`.

The Matlab files currently contain several generation blocks, most of which are commented. Select the block you need, update the save path if necessary, and run:

```bash
matlab -batch "run('SSA/Ex42Vilar2002R.m')"
```

or

```bash
matlab -batch "run('SSA/Ex41SSACIRC73s.m')"
```

The current Matlab files use hard-coded output paths under `/Users/jesse/Dropbox/DataProd/`. Change these paths before running on another machine.

## Output Data Format

Most generated `.mat` files contain a variable named `data`.

For SDE/SPDE examples, the standard shape is:

```text
data.shape = [dim, Nt + 1, N_data]
```

where:

- `dim` is the state dimension;
- `Nt + 1` is the number of saved time points;
- `N_data` is the number of trajectories or trajectory pairs.

For one-dimensional examples, `dim = 1`.

Controlled or nonautonomous examples often also save:

```text
para.shape = [dim_para, Nt, N_data]
```

where `para` stores the local control/excitation parameters used over each time step.

Conditional SSA files often use dictionary-style variables:

```text
0_i, 0_d, 1_i, 1_d, ...
```

where `*_i` stores the fixed initial condition and `*_d` stores samples from the corresponding conditional distribution.

Original SSA path files may store variables such as:

```text
t_0, d_0, t_1, d_1, ...
```

where `t_*` is the reaction time sequence and `d_*` is the corresponding population path.
