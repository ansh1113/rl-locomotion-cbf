# Control Barrier Functions: Theory and Implementation

## Introduction

Control Barrier Functions (CBFs) provide a mathematical framework for ensuring safety in dynamical systems. This document explains the theory behind CBFs and how they are implemented in this project.

## Mathematical Background

### Barrier Functions

A **barrier function** $h: \mathbb{R}^n \rightarrow \mathbb{R}$ defines a safe set:

$$\mathcal{C} = \{x \in \mathbb{R}^n : h(x) \geq 0\}$$

When $h(x) \geq 0$, the state $x$ is considered safe. When $h(x) < 0$, the state is unsafe.

### Control Barrier Functions

For a control-affine system:

$$\dot{x} = f(x) + g(x)u$$

A function $h(x)$ is a **Control Barrier Function** if there exists a class-$\mathcal{K}$ function $\alpha$ such that:

$$\sup_u \left[\dot{h}(x, u)\right] = \sup_u \left[L_f h(x) + L_g h(x) \cdot u\right] \geq -\alpha(h(x))$$

where:
- $L_f h = \nabla h \cdot f(x)$ is the Lie derivative of $h$ along $f$
- $L_g h = \nabla h \cdot g(x)$ is the Lie derivative of $h$ along $g$
- $\alpha: \mathbb{R} \rightarrow \mathbb{R}$ is a class-$\mathcal{K}$ function (e.g., $\alpha(h) = \lambda h$)

### Safety Guarantee

If the CBF condition holds, then:
1. If $h(x_0) \geq 0$ (initially safe)
2. Then $h(x_t) \geq 0$ for all $t \geq 0$ (remain safe forever)

This provides a **mathematical guarantee** of safety.

## Implementation

### CBF-QP Formulation

To filter unsafe actions from an RL policy, we solve a Quadratic Program:

$$
\begin{align}
\min_{u} \quad & \|u - u_{\text{des}}\|^2 + \rho s^2 \\
\text{s.t.} \quad & L_f h_i(x) + L_g h_i(x) \cdot u + \alpha(h_i(x)) \geq -s \\
& u_{\min} \leq u \leq u_{\max} \\
& s \geq 0
\end{align}
$$

where:
- $u_{\text{des}}$ is the desired action from the PPO policy
- $s$ is a slack variable for constraint relaxation
- $\rho$ is the penalty weight for slack
- $h_i$ are multiple barrier functions

### Barrier Functions Used

#### 1. Stability Barrier

Ensures the Center of Mass (CoM) stays within the support polygon:

$$h_{\text{stab}}(x) = d(\text{CoM}_{xy}, \partial P) - \epsilon$$

where:
- $\text{CoM}_{xy}$ is the CoM projection on ground
- $P$ is the support polygon from foot contacts
- $d(\cdot, \partial P)$ is distance to polygon boundary
- $\epsilon$ is a safety margin

#### 2. Height Barrier

Ensures minimum body height:

$$h_{\text{height}}(x) = z - z_{\min}$$

where:
- $z$ is the body height
- $z_{\min}$ is the minimum safe height (e.g., 0.2m)

#### 3. Joint Limit Barriers

Ensures joint positions stay within limits:

$$h_{i}^{\text{lower}}(x) = q_i - q_{i,\min} - \epsilon$$
$$h_{i}^{\text{upper}}(x) = q_{i,\max} - q_i - \epsilon$$

for each joint $i$.

### Class-K Functions

We use a simple linear class-$\mathcal{K}$ function:

$$\alpha(h) = \lambda h$$

where $\lambda > 0$ is a tunable parameter. Higher values make the system more conservative (safer but potentially slower).

## Computational Efficiency

### QP Solver

We use OSQP (Operator Splitting QP), which provides:
- Fast solve times (~2-3ms)
- Warm starting for sequential problems
- Robust handling of infeasibility

### Lie Derivative Computation

Lie derivatives are computed using finite differences:

$$\nabla h \approx \frac{h(x + \epsilon e_i) - h(x)}{\epsilon}$$

for each basis vector $e_i$.

## Safety vs Performance Trade-off

The CBF parameter $\alpha$ controls the trade-off:

- **Low $\alpha$**: Less conservative, faster motion, but less safety margin
- **High $\alpha$**: More conservative, slower motion, but larger safety margin

Typical values: $\alpha \in [0.5, 2.0]$

## Extensions

### Multiple CBFs

Multiple barrier functions are combined using:

$$\dot{h}_i + \alpha_i(h_i) \geq 0 \quad \forall i$$

Each becomes a separate constraint in the QP.

### High Order CBFs

For systems with relative degree > 1, we use higher-order CBFs:

$$h^{(k)} + \alpha_{k-1}(h^{(k-1)}) \geq 0$$

## References

1. Ames, A. D., et al. "Control barrier functions: Theory and applications." IEEE CDC, 2019.
2. Xu, X., et al. "Robustness of control barrier functions for safety critical control." IFAC, 2015.
3. Cheng, X., et al. "End-to-end safe reinforcement learning through barrier functions for safety-critical continuous control tasks." AAAI, 2019.
