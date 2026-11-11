"""Global hyperparameters for Minesweeper DQN."""
from dataclasses import dataclass
 
 
@dataclass(frozen=True)
class Config:
    # --- environment -------------------------------------------------------
    width: int = 8
    height: int = 8
    num_mines: int = 10
 
    # --- network -----------------------------------------------------------
    hidden_channels: int = 64
    num_conv_layers: int = 6  # 3x3 convs; receptive field grows with depth
 
    # --- DQN ---------------------------------------------------------------
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 128
    buffer_capacity: int = 100_000
    learning_starts: int = 2_000  # transitions collected before updates begin
    train_freq: int = 1  # env steps per gradient step
    target_update_every: int = 1_000  # hard sync theta- <- theta every N gradient steps
    grad_clip: float = 10.0
 
    # --- epsilon-greedy (linear decay) ---------------------------------------
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
 
    # --- run control -------------------------------------------------------
    total_steps: int = 500_000
    eval_every: int = 10_000
    eval_episodes: int = 200
    log_every: int = 1_000
    seed: int = 0
    log_dir: str = "runs/dqn_minesweeper"
    checkpoint_dir: str = "checkpoints"
 
 
CONFIG = Config()