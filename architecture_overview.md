# Minesweeper RL: System Architecture & MDP Overview (DQN)

This document outlines the formal Markov Decision Process (MDP) formulation, spatial tensor geometry, masked Q-learning mechanics, and software component responsibilities for training a Deep Q-Network (DQN) agent to play Minesweeper.

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

*Note: One-hot state encoding prevents artificial linear numerical bias between categorical tile states.*

### Action Space ($\mathcal{A}$)
* **Type:** Discrete space of size $|\mathcal{A}| = H \times W$.
* **Overloaded Spatial Targeting:** Action $a_t \in [0, H \times W - 1]$ maps directly to grid coordinate $(r, c) = (\lfloor a / W \rfloor, a \bmod W)$:
  * **Target cell is Covered:** Executes a standard **Reveal** action (with first-click mine safety).
  * **Target cell is Revealed ($1 \le k \le 8$):** Executes a **Chord** action (auto-opens adjacent unflagged tiles if surrounding flag count equals $k$).

### Action Masked Q-Selection ($\mathcal{M}$)
To prevent illegal moves during Q-value selection, a binary action mask $M_t \in \{0, 1\}^{H \times W}$ is calculated at each step. Invalid actions ($M(s, a) = 0$) are assigned a $Q$-value of $-\infty$:

$$\hat{Q}(s, a) = \begin{cases} Q(s, a) & \text{if } M(s, a) = 1 \\ -\infty & \text{if } M(s, a) = 0 \end{cases}$$

* **Exploitation Action:** $a_t = \arg\max_{a} \hat{Q}(s_t, a)$
* **$\epsilon$-Greedy Exploration:** Action $a_t$ is sampled uniformly at random strictly from valid mask indices $\{a \mid M(s, a) = 1\}$.

### Reward Function ($\mathcal{R}$)
Dense step-wise feedback balances standard safe reveals and risky guesses:

$$R_t = \begin{cases} 
+0.1 \times m & \text{if } m \text{ safe tiles are successfully revealed (single click or chord)} \\
-1.0 & \text{if a mine is detonated (terminal loss state)} \\
+1.0 + (0.1 \times m) & \text{if all non-mine safe tiles are cleared (terminal win state)}
\end{cases}$$

---

## 2. DQN Training & Loss Mechanics

### Bellman Target Equation (Double DQN)
To reduce Q-value overestimation bias inherent under partial observability, Double DQN is used for parameter updates:

$$Y_t = R_{t+1} + \gamma \cdot Q\left(s_{t+1}, \arg\max_a \hat{Q}(s_{t+1}, a; \theta); \theta^-\right)$$

where $\theta$ represents online Q-network parameters and $\theta^-$ represents periodically synchronized target network parameters.

---

## 3. Codebase Architecture & File Responsibilities