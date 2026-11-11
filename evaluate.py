"""Deterministic (epsilon = 0) evaluation: win rate and step efficiency."""
import argparse
 
import numpy as np
import torch
 
from config import Config
from minesweeper_env import MinesweeperEnv
from model import MinesweeperDQN, masked_argmax
 
 
@torch.no_grad()
def evaluate_policy(
    net: MinesweeperDQN,
    cfg: Config,
    n_episodes: int = 200,
    seed: int = 10_000,
    device: str | torch.device = "cpu",
) -> dict:
    """Greedy masked rollouts on boards seeded seed..seed+n_episodes-1, so the
    same set of test boards is reused across calls."""
    was_training = net.training
    net.eval()
    env = MinesweeperEnv(cfg.width, cfg.height, cfg.num_mines)
 
    wins, steps, rewards, cleared = [], [], [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        done, ep_reward, n_steps = False, 0.0, 0
        while not done:
            x = torch.as_tensor(obs, device=device).unsqueeze(0)
            m = torch.as_tensor(info["action_mask"], device=device).unsqueeze(0)
            action = int(masked_argmax(net(x), m).item())
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += r
            n_steps += 1
        wins.append(env.won)
        steps.append(n_steps)
        rewards.append(ep_reward)
        cleared.append(env.revealed_safe / env.total_safe)
 
    net.train(was_training)
    wins_arr, steps_arr = np.array(wins), np.array(steps)
    return {
        "win_rate": float(wins_arr.mean()),
        "mean_steps": float(steps_arr.mean()),
        "mean_steps_on_wins": float(steps_arr[wins_arr].mean()) if wins_arr.any() else float("nan"),
        "mean_reward": float(np.mean(rewards)),
        "mean_cleared_fraction": float(np.mean(cleared)),
    }
 
 
def load_checkpoint(path: str, device: str = "cpu") -> tuple[MinesweeperDQN, Config]:
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["config"])
    net = MinesweeperDQN(11, cfg.hidden_channels, cfg.num_conv_layers).to(device)
    net.load_state_dict(ckpt["model"])
    return net, cfg
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=10_000)
    args = p.parse_args()
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = load_checkpoint(args.checkpoint, device)
    stats = evaluate_policy(net, cfg, args.episodes, args.seed, device)
    print(f"Board {cfg.width}x{cfg.height}, {cfg.num_mines} mines, {args.episodes} episodes")
    for k, v in stats.items():
        print(f"  {k:24s} {v:.4f}")