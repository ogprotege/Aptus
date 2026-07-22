# Memory Estimation

Methodology version: `aptus-memory-v2`.

Aptus v0.2 emits a point estimate and a heuristic upper planning envelope. The
upper value is not a proven bound or statistical confidence interval. No
calibration database supports either claim.

## Units

All internal memory values use bytes. Displayed GiB uses:

$$
1\ \mathrm{GiB} = 2^{30}\ \mathrm{bytes}
$$

## Per-device equation

For device $d$, component $i$ has point value $p_{i,d}$ and upper value
$u_{i,d}$, where $u_{i,d} \ge p_{i,d}$.

Components are:

- resident base weights $W$;
- quantization metadata $Q$;
- trainable parameters $A$;
- gradients $G$;
- optimizer states and master weights $O$;
- saved activations $X$;
- communication buffers $C$;
- kernel workspaces $K$;
- load and checkpoint transients $T$;
- allocator and fragmentation allowance $F$.

The envelopes are:

$$
M_{\mathrm{point},d} =
W_d + Q_d + A_d + G_d + O_d + X_d + C_d + K_d + T_d + F_d
$$

$$
M_{\mathrm{upper},d} = \left\lceil
u_{W,d} + u_{Q,d} + u_{A,d} + u_{G,d} + u_{O,d} + u_{X,d} +
u_{C,d} + u_{K,d} + u_{T,d} + u_{F,d}
\right\rceil
$$

The plan records every term. It does not hide a global percentage inside the
total.

## Exact state terms

For $N$ resident values stored at $b$ bits:

$$
M_{\mathrm{values}} = \left\lceil \frac{Nb}{8} \right\rceil
$$

An estimator may derive quantization scales, zero points, and nested metadata
from exact group sizes and metadata dtypes. The v0.2 estimator instead uses the
explicit analytical coefficients below.

For trainable parameter count $P_t$:

$$
M_A = P_t \times \operatorname{bytes}(\mathrm{adapter\ dtype})
$$

$$
M_G = P_t \times \operatorname{bytes}(\mathrm{gradient\ dtype})
$$

For optimizer state tensors $s \in S$:

$$
M_O = P_t \times \sum_{s \in S}\operatorname{bytes}(s)
      + M_{\mathrm{master\ weights}}
$$

The v0.2 compiler uses the optimizer-state coefficient below. A future compiler
must version any additional optimizer policy and its state tensors.

## V0.2 state coefficients

The current v0.2 rule set uses these explicit analytical priors:

| Term | Point rule |
| --- | --- |
| Full or LoRA base weights | $2P/s$ bytes |
| 8-bit LoRA base weights | $P/s$ bytes |
| QLoRA base weights | $0.5P/s$ bytes |
| 8-bit metadata | $0.05P/s$ bytes |
| QLoRA metadata | $P(0.127/8)/s$ bytes |
| Full gradients | $2P_t/s$ bytes |
| Adapter gradients | $4P_t/s$ bytes |
| Adapter weights | $4P_t/s$ bytes |
| Optimizer states | $8P_t/s$ bytes |

Here $P$ is total parameters, $P_t$ is trainable parameters, and $s$ is the
state-sharding divisor. For single and DDP, $s=1$. The simplified v0.2 FSDP
prior sets $s$ to world size for the listed state terms. This division does not
model an exact FSDP schedule.

These coefficients describe the generated v0.2 trainer. They are not universal
constants for other optimizers, FSDP policies, or libraries.

For adapter methods, v0.2 estimates trainable parameters from the catalog
module names. Per layer, `q_proj`, `k_proj`, `v_proj`, and `o_proj` each add
$2hr$ parameters. `gate_proj`, `up_proj`, and `down_proj` each add
$(h+i)r$, where $h$ is hidden size, $i$ is intermediate size, and $r$ is rank.
When intermediate size is absent, the estimator uses $i=4h$.

## Activation prior

Before measurement, v0.2 uses this explicit activation prior:

$$
X_{\mathrm{point}} =
b \times q \times h \times L \times 2 \times 2.5
$$

Here $b$ is per-device micro-batch, $q$ is sequence length, $h$ is hidden size,
$L$ is model layer count, 2 is the compute-byte prior, and 2.5 is the v0.2
activation factor. Gradient checkpointing is assumed.

The upper activation value is:

$$
u_X = \left\lceil 1.35 X_{\mathrm{point}} \right\rceil
$$

The plan records both coefficients and evidence IDs. An unsupported model or
runtime causes abstention, not a guessed Llama default.

## V0.2 overhead equations

Let $WQ=W+Q$. The point rules are:

$$
K = \max(0.5\ \mathrm{GiB},\ 0.02WQ)
$$

$$
T_{temporary} = \max(0.5\ \mathrm{GiB},\ 0.04WQ)
$$

$$
T_{load} = 0.20WQ
$$

Let $M_{before\ allocator}$ be the sum of all point components before allocator
allowance. Then:

$$
F = 0.08M_{before\ allocator}
$$

DDP communication is:

$$
C_{DDP} = \min(2P_t,\ 2\ \mathrm{GiB})
$$

The supported FSDP planning prior is:

$$
C_{FSDP} = \min\left(\frac{2P_t}{w},\ 3\ \mathrm{GiB}\right)
$$

where $w$ is world size. Single-device communication is zero.

The serialized heuristic upper components retain unchanged deterministic state
terms and apply:

- activations: $1.35X$;
- communication and load transient: $1.25C$ and $1.25T_{load}$;
- workspace, temporary, and allocator: $1.50K$, $1.50T_{temporary}$,
  and $1.50F$;
- uncertainty: $0.10M_{point}$.

Therefore:

$$
M_{upper} = \sum_i u_i
$$

The serialized `component_upper_bounds` map contains every $u_i$. The upper
total must equal their sum. The user reserve is not a usage component.

## Device budget and fit

For total VRAM $V_d$, user reserve $R_d$, and optional measured free VRAM
$V_d^{free}$, define available VRAM as:

$$
B_d = \max\left(0,
\begin{cases}
V_d^{free}-R_d, & \text{when measured free VRAM exists} \\
V_d-R_d, & \text{otherwise}
\end{cases}
\right)
$$

Load and runtime transients are already named usage components in the point and
upper envelopes. The user reserve remains a separate reduction in the fit
budget.

A candidate predicts fit only if:

$$
M_{\mathrm{upper},d} \le B_d \quad \text{for every selected device } d
$$

Predicted fit still requires an exact real-model pilot. Synthetic measured
preflight checks the selected method and kernel, not planned-model fit.

## Host RAM and staging-disk checks

V0.2 applies two additional planning rules. Let $L=1$ for a single-device
candidate and $L=w$ for DDP or FSDP, because each launched rank can stage a
model copy on the host. Then:

$$
H_{required}=2.2PL
$$

Let $T$ be the estimated number of trainable parameters and let:

$$
U_{checkpoint}=\begin{cases}
10T, & \text{full fine-tuning} \\
12T, & \text{adapter method}
\end{cases}
$$

The disk rule is:

$$
D_{required}=2.2P+D_{source}+D_{canonical}+D_{pilot}+3U_{checkpoint}
+4U_{checkpoint}+E_{final}
$$

Here $P$ is model parameter count, used directly as a byte coefficient.
$D_{source}$ and $D_{canonical}$ are the source and compiled canonical dataset
sizes. $D_{pilot}$ is the largest canonical row size multiplied by
$\max(32,2B_{effective})$. The planner retains three checkpoint units and allows
four more for the two-phase pilot workspace. $E_{final}$ is the larger of 0.0625
GiB or two bytes per trainable parameter for full training and four bytes per
trainable parameter for adapter export.

These remain uncalibrated heuristics. The host check uses free host RAM when
present, otherwise total host RAM. The disk check runs only when free disk was
supplied or measured. The disk value covers the named staging, dataset, bounded
pilot, retained-checkpoint, and final-export allowances. It does not cover
unbounded logs, package caches, provider cache transients, or unrelated files.

## Distribution

Single-device and DDP candidates carry full replicated state unless a component
rule states otherwise. DDP never sums VRAM across devices.

The v0.2 FSDP prior divides base, quantization, adapter, gradient, and optimizer
state terms by world size. It adds the capped communication term shown above.
It does not separately model wrapping boundaries, prefetch, all-gather,
reduction, or checkpoint spikes. Aptus rejects full-parameter FSDP in v0.2.
LoRA FSDP remains conditional even when this heuristic envelope fits. It uses
`use_orig_params=true` for mixed frozen and trainable parameters, which can
increase gradient memory beyond the simplified sharding prior. The exact
generated wrapping path must pass a real-model pilot.

## Known limits

The analytical activation, workspace, transient, fragmentation, and FSDP terms
are uncalibrated priors. Their upper values are planning allowances. See
[preflight and calibration](preflight-calibration.md).
