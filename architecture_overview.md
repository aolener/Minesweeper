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
The codebase is structured modularly to decouple game physics, model training, policy evaluation, and global hyperparameter settings.

minesweeper-rl/
├── architecture_overview.md  # System MDP design & DQN architectural specs
├── config.py                 # Global hyperparameters (learning rate, buffer size, gamma, epsilon decay)
├── evaluate.py               # Evaluation loops and win-rate analytics
├── minesweeper_env.py        # Custom Gymnasium environment implementation
├── model.py                  # PyTorch Deep Q-Network CNN architecture
├── requirements.txt          # Python environment dependencies
└── train.py                  # Replay buffer management, DQN loss update, and target net syncing

### Modular File Definitions

#### `minesweeper_env.py`
* **Domain:** Custom Gymnasium Environment (`MinesweeperEnv`).
* **Responsibilities:**
  * Tracks internal solution layout (`board`), visible state (`visible`), and flags (`flags`).
  * Implements first-click safety mine generation.
  * Handles recursive zero-cascade flood fills (BFS queue) when revealing empty $0$ tiles.
  * Resolves chord mechanics and surrounding tile reveals.
  * Generates $11 \times H \times W$ observation tensors and computes `action_masks()`.

#### `model.py`
* **Domain:** Q-Network Architecture.
* **Responsibilities:**
  * Fully Convolutional Network (FCN) mapping $(11, H, W) \to Q(s) \in \mathbb{R}^{H \times W}$.
  * Preserves spatial dimensions using $3 \times 3$ padded convolutions with $1 \times 1$ conv final output layer.

#### `config.py`
* **Domain:** Hyperparameters.
* **Responsibilities:**
  * Replay buffer capacity (e.g., $100,000$ transitions), mini-batch size (e.g., $64$ or $128$), discount factor $\gamma = 0.99$, learning rate $\alpha$, $\epsilon$-decay schedule, and target network update frequency ($\tau$).

#### `train.py`
* **Domain:** DQN Training Pipeline.
* **Responsibilities:**
  * Manages Experience Replay Buffer $\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, d_t, M_{t+1})\}$.
  * Runs $\epsilon$-greedy action selection with valid masking.
  * Computes Huber loss (Smooth L1 Loss) between predicted $Q(s, a)$ and Double DQN targets $Y_t$.
  * Syncs target network weights ($\theta^- \leftarrow \theta$) and logs metrics to TensorBoard.

#### `evaluate.py`
* **Domain:** Benchmarking.
* **Responsibilities:**
  * Runs deterministic evaluation ($\epsilon = 0.0$) across test boards to record win rates and step efficiency.