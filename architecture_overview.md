# Minesweeper RL: System Architecture & MDP Overview

This document outlines the formal Markov Decision Process (MDP) formulation, state space tensor geometry, action masking mechanics, and software component responsibilities for training a Reinforcement Learning (RL) agent to play Minesweeper.

---

## 1. MDP Formulation

Minesweeper is modeled as a 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ operating as a Partially Observable Markov Decision Process (POMDP).

### State Space ($\mathcal{S}$)
The observation state $s_t \in \mathbb{R}^{11 \times H \times W}$ is an 11-channel binary spatial feature tensor representing an $H \times W$ grid:

| Channel Index | Binary Feature Representation |
| :--- | :--- |
| **Channels 0–8** | One-hot indicator grids for revealed numbers $0$ through $8$. |
| **Channel 9** | Binary mask for unrevealed / covered tiles ($1$ = covered, $0$ = revealed). |
| **Channel 10** | Binary mask for flagged tiles ($1$ = flagged, $0$ = unflagged). |

*Note: One-hot state encoding prevents artificial linear bias between categorical states (e.g., preventing the network from treating an unrevealed tile as numerically between a 1 and a 2).*

### Action Space ($\mathcal{A}$)
* **Type:** Discrete space of size $|\mathcal{A}| = H \times W$.
* **Overloaded Spatial Targeting:** Action $a_t \in [0, H \times W - 1]$ maps directly to grid coordinate $(r, c) = (\lfloor a / W \rfloor, a \bmod W)$:
  * **Target cell is Covered:** Executes a standard **Reveal** action (with first-click mine safety).
  * **Target cell is Revealed ($1 \le k \le 8$):** Executes a **Chord** action (auto-opens adjacent unflagged tiles if surrounding flag count equals $k$).

### Action Masking ($\mathcal{M}$)
To prevent invalid or redundant moves, a binary action mask $M_t \in \{0, 1\}^{H \times W}$ is computed at each timestep. Masked actions receive an additive logit penalty of $-\infty$ prior to policy sampling:

$$M(r, c) = \begin{cases} 
1 & \text{if tile } (r, c) \text{ is covered (valid reveal)} \\
1 & \text{if tile } (r, c) \text{ is revealed with number } k \text{ AND } \text{flags}_{\text{neighbors}} = k \\
0 & \text{otherwise (illegal move)}
\end{cases}$$

### Reward Function ($\mathcal{R}$)
Dense step-wise feedback balances standard safe reveals and risky guesses:

$$R_t = \begin{cases} 
+0.1 \times m & \text{if } m \text{ safe tiles are successfully revealed (single click or chord)} \\
-1.0 & \text{if a mine is detonated (terminal loss state)} \\
+1.0 + (0.1 \times m) & \text{if all non-mine safe tiles are cleared (terminal win state)}
\end{cases}$$

---

## 2. Codebase Architecture & File Responsibilities

The codebase is structured modularly to decouple game physics, model training, policy evaluation, and global hyperparameter settings.