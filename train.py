"""Double-DQN training with masked action selection and a live matplotlib window.
 
    python3 train.py                          # train; a window shows the agent playing
    python3 train.py --no-viz                 # train without the window
    python3 train.py --replay checkpoints/best.pt   # just watch a saved agent
 
Training runs in a background thread; the window (matplotlib, main thread) replays
games from the current agent on the same fixed boards, so you can watch it improve.
"""
import argparse
import os
import random
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
 
import numpy as np
import torch
import torch.nn.functional as F
 
from config import CONFIG, Config
from evaluate import evaluate_policy, load_checkpoint
from minesweeper_env import MinesweeperEnv
from model import MinesweeperDQN, mask_q, masked_argmax
 
 
# =============================================================== replay buffer
class ReplayBuffer:
    """Stores D = (s, a, r, s', d, M') as preallocated arrays (obs are binary -> uint8)."""
 
    def __init__(self, capacity: int, obs_shape: tuple, n_actions: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=bool)
        self.idx = 0
        self.size = 0
 
    def add(self, obs, action, reward, next_obs, done, next_mask) -> None:
        i = self.idx
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.next_masks[i] = next_mask
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
 
    def __len__(self) -> int:
        return self.size
 
    def sample(self, batch_size: int, device) -> tuple[torch.Tensor, ...]:
        ix = np.random.randint(0, self.size, size=batch_size)
        t = lambda a, dt: torch.as_tensor(a[ix], dtype=dt, device=device)
        return (
            t(self.obs, torch.float32),
            t(self.actions, torch.int64),
            t(self.rewards, torch.float32),
            t(self.next_obs, torch.float32),
            t(self.dones, torch.float32),
            t(self.next_masks, torch.bool),
        )
 
 
# ============================================================ DQN core pieces
def epsilon_at(step: int, cfg: Config) -> float:
    frac = min(1.0, step / cfg.eps_decay_steps)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
 
 
def select_action(net, obs, mask, eps, device) -> int:
    """Epsilon-greedy restricted to legal actions."""
    if random.random() < eps:
        return int(np.random.choice(np.flatnonzero(mask)))
    with torch.no_grad():
        x = torch.as_tensor(obs, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, device=device).unsqueeze(0)
        return int(masked_argmax(net(x), m).item())
 
 
def dqn_loss(online, target, batch, gamma: float) -> torch.Tensor:
    s, a, r, s2, d, m2 = batch
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN: online net picks the (masked) best action, target net scores it.
        a_star = mask_q(online(s2), m2).argmax(dim=1, keepdim=True)
        q_next = target(s2).gather(1, a_star).squeeze(1)
        y = r + gamma * (1.0 - d) * q_next  # terminal rows ignore q_next
    return F.smooth_l1_loss(q, y)  # Huber
 
 
def save_checkpoint(path: str, net, cfg: Config, **extra) -> None:
    torch.save({"model": net.state_dict(), "config": asdict(cfg), **extra}, path)
 
 
# ================================================== recording games for the UI
def net_policy(net, device="cpu"):
    """Greedy masked policy: policy(obs, mask) -> (action, list of masked Q-values)."""
 
    @torch.no_grad()
    def policy(obs, mask):
        x = torch.as_tensor(obs, device=device).unsqueeze(0)
        m = torch.as_tensor(mask, device=device).unsqueeze(0)
        q = net(x)
        action = int(mask_q(q, m).argmax(dim=1).item())
        qs = [round(float(v), 3) if ok else None for v, ok in zip(q[0].tolist(), mask)]
        return action, qs
 
    return policy
 
 
def _snapshot(env, last, reward, q):
    return {
        "visible": env.visible.tolist(),
        "flags": env.flags.astype(int).tolist(),
        "last": last,  # [row, col, "reveal"/"chord"] of the move that produced this state
        "reward": float(reward),
        "revealed": int(env.revealed_safe),
        "q": q,  # agent's Q-values at this state (None for illegal tiles / final state)
    }
 
 
def play_recorded_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    action, q = policy(obs, info["action_mask"])
    frames = [_snapshot(env, None, 0.0, q)]
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        last = [action // env.width, action % env.width, info["action_type"]]
        if done:
            action, q = None, None
        else:
            action, q = policy(obs, info["action_mask"])
        frames.append(_snapshot(env, last, reward, q))
    return {
        "seed": seed,
        "frames": frames,
        "mines": [tuple(m) for m in np.argwhere(env.board).tolist()],
        "detonated": [tuple(map(int, t)) for t in env.detonated],
        "won": bool(env.won),
        "lost": bool(env.lost),
    }
 
 
def build_replay(env, policy, n_episodes, seed, label, version) -> dict:
    return {
        "label": label,
        "version": version,
        "total_safe": env.total_safe,
        "mines": env.num_mines,
        "episodes": [play_recorded_episode(env, policy, seed + i) for i in range(n_episodes)],
    }
 
 
class SharedState:
    """Thread-safe mailbox between the training thread and the window."""
 
    def __init__(self):
        self.lock = threading.Lock()
        self.replay = None  # latest replay dict (never mutated after being set)
        self.evals = []  # (step, eval win rate, eval cleared fraction)
        self.status = "Starting..."
        self.stop = threading.Event()
        self.finished = False
 
    def set_replay(self, data):
        with self.lock:
            self.replay = data
 
    def add_eval(self, step, win, cleared):
        with self.lock:
            self.evals.append((step, win, cleared))
 
    def set_status(self, text):
        with self.lock:
            self.status = text
 
 
# ============================================================== matplotlib UI
class ReplayWindow:
    COV, REV, NEW, BOOM = "#c7cdd4", "#fafaf9", "#fde68a", "#ef4444"
    NUM = {1: "#2563eb", 2: "#15803d", 3: "#dc2626", 4: "#1e3a8a",
           5: "#92400e", 6: "#0f766e", 7: "#44403c", 8: "#78716c"}
 
    def __init__(self, shared, width, height, total_steps=None, live=True):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.widgets import Button, CheckButtons, Slider
 
        self.plt, self.shared, self.W, self.H, self.live = plt, shared, width, height, live
        self.data = None
        self.pending = None
        self.ep, self.fi, self.held = 0, 0, 0
        self.playing = True
        self.show_q = False
        self.evals_seen = 0
        self.last_status = None
        self.cmap = plt.get_cmap("Blues")
        self.fs = max(5.0, min(15.0, 110.0 / max(width, height)))
 
        fig = self.fig = plt.figure(figsize=(12, 7.5))
        if getattr(fig.canvas, "manager", None) is not None:
            fig.canvas.manager.set_window_title("Minesweeper DQN")
 
        # ---- board
        ax = self.ax = fig.add_axes([0.03, 0.20, 0.52, 0.76])
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.axis("off")
        self.cells = []
        for r in range(height):
            for c in range(width):
                x, y = c, height - 1 - r
                rect = Rectangle((x, y), 1, 1, facecolor=self.COV, edgecolor="#8a8f98", lw=1.2)
                ax.add_patch(rect)
                dot = Circle((x + 0.5, y + 0.5), 0.24, facecolor="#1f2937", edgecolor="none",
                             zorder=3, visible=False)
                ax.add_patch(dot)
                ring = Rectangle((x, y), 1, 1, fill=False, lw=3, zorder=5, visible=False)
                ax.add_patch(ring)
                txt = ax.text(x + 0.5, y + 0.5, "", ha="center", va="center", zorder=4,
                              fontsize=self.fs, fontweight="bold")
                self.cells.append((rect, txt, dot, ring))
 
        # ---- text panel
        ax_t = fig.add_axes([0.58, 0.62, 0.40, 0.34])
        ax_t.axis("off")
        ax_t.text(0.0, 1.0, "Minesweeper DQN agent", fontsize=16, fontweight="bold", va="top")
        self.sub_txt = ax_t.text(0.0, 0.86, "", fontsize=10.5, va="top", color="#555", style="italic")
        self.info_txt = ax_t.text(0.0, 0.70, "", fontsize=12, va="top", linespacing=1.55)
        self.status_txt = ax_t.text(0.0, 0.18, "", fontsize=13, va="top", fontweight="bold")
        self.train_txt = ax_t.text(0.0, 0.04, "", fontsize=9.5, va="top", color="#555")
 
        # ---- progress plot (training only)
        self.ax_prog = fig.add_axes([0.62, 0.27, 0.35, 0.27])
        if live:
            self.ax_prog.set_title("Greedy evaluation during training", fontsize=10)
            self.ax_prog.set_xlabel("training step", fontsize=9)
            self.ax_prog.set_ylim(0, 1.02)
            self.ax_prog.set_xlim(0, total_steps or 1)
            self.ax_prog.grid(alpha=0.3)
            self.ax_prog.tick_params(labelsize=8)
            (self.l_win,) = self.ax_prog.plot([], [], "o-", color="#15803d", label="win rate", ms=3)
            (self.l_clr,) = self.ax_prog.plot([], [], "o-", color="#2563eb", label="tiles cleared", ms=3)
            self.ax_prog.legend(loc="upper left", fontsize=8)
        else:
            self.ax_prog.axis("off")
 
        # ---- controls
        def button(rect, label, cb, color="#e5e7eb"):
            b = Button(fig.add_axes(rect), label, color=color, hovercolor="#d1d5db")
            b.on_clicked(cb)
            return b
 
        self.b_back = button([0.03, 0.09, 0.07, 0.065], "< Back", lambda e: self._step(-1))
        self.b_play = button([0.11, 0.09, 0.09, 0.065], "Pause", lambda e: self._set_playing(not self.playing))
        self.b_next = button([0.21, 0.09, 0.07, 0.065], "Next >", lambda e: self._step(+1))
        self.b_game = button([0.29, 0.09, 0.10, 0.065], "Next game", lambda e: self._next_game())
        self.b_load = button([0.40, 0.09, 0.19, 0.065], "Agent is up to date", lambda e: self._load_pending())
        self.b_stop = None
        if live:
            self.b_stop = button([0.60, 0.09, 0.13, 0.065], "Stop training", self._on_stop, "#fecaca")
        self.s_speed = Slider(fig.add_axes([0.10, 0.03, 0.28, 0.025]), "Delay (ms)", 20, 1000,
                              valinit=350, valstep=10)
        self.s_speed.on_changed(self._on_speed)
        self.chk = CheckButtons(fig.add_axes([0.78, 0.07, 0.19, 0.08]), ["Show Q-values"], [False])
        self.chk.on_clicked(self._on_q)
        fig.text(0.03, 0.012, "Blue ring = last move (orange = chord)  |  yellow = newly revealed  |  "
                 "dashed ring = agent's next pick (Q-values on)  |  arrow keys step, space plays/pauses",
                 fontsize=8, color="#777", va="center")
        fig.canvas.mpl_connect("key_press_event", self._on_key)
 
        self.anim = FuncAnimation(fig, self._tick, interval=int(self.s_speed.val),
                                  blit=False, cache_frame_data=False)
 
    # ------------------------------------------------------------ controls
    def show(self):
        self.plt.show()
 
    def _set_playing(self, on):
        self.playing = on
        self.b_play.label.set_text("Pause" if on else "Play")
        self.fig.canvas.draw_idle()
 
    def _on_speed(self, v):
        try:
            self.anim.event_source.interval = int(v)
        except Exception:
            pass
 
    def _on_q(self, _label):
        self.show_q = self.chk.get_status()[0]
        if self.data is not None:
            self._draw()
            self.fig.canvas.draw_idle()
 
    def _on_stop(self, _e):
        self.shared.stop.set()
        self.b_stop.label.set_text("Stopping...")
        self.fig.canvas.draw_idle()
 
    def _on_key(self, e):
        if e.key == "right":
            self._step(+1)
        elif e.key == "left":
            self._step(-1)
        elif e.key == " ":
            self._set_playing(not self.playing)
 
    def _forward(self):
        E = self.data["episodes"][self.ep]
        if self.fi < len(E["frames"]) - 1:
            self.fi += 1
            return True
        if self.ep < len(self.data["episodes"]) - 1:
            self.ep += 1
            self.fi = 0
            return True
        return False
 
    def _step(self, d):
        if self.data is None:
            return
        if d > 0:
            self._forward()
        elif self.fi > 0:
            self.fi -= 1
        elif self.ep > 0:
            self.ep -= 1
            self.fi = len(self.data["episodes"][self.ep]["frames"]) - 1
        self._draw()
        self.fig.canvas.draw_idle()
 
    def _next_game(self):
        if self.data is not None and self.ep < len(self.data["episodes"]) - 1:
            self.ep += 1
            self.fi = 0
            self.held = 0
            self._draw()
            self.fig.canvas.draw_idle()
 
    def _load_pending(self):
        if self.pending is not None:
            self._apply(self.pending)
            self.fig.canvas.draw_idle()
 
    def _set_load_button(self):
        if self.pending is not None:
            self.b_load.label.set_text(f"Load newer agent (step {self.pending['version']:,})")
            self.b_load.ax.set_facecolor("#bfdbfe")
            self.b_load.color = "#bfdbfe"
        else:
            self.b_load.label.set_text("Agent is up to date")
            self.b_load.ax.set_facecolor("#e5e7eb")
            self.b_load.color = "#e5e7eb"
 
    # ------------------------------------------------------------ animation
    def _apply(self, rep):
        self.data = rep
        self.ep, self.fi, self.held = 0, 0, 0
        self.pending = None
        self._set_load_button()
        wins = sum(e["won"] for e in rep["episodes"])
        self.sub_txt.set_text(f"{rep['label']}   |   {wins}/{len(rep['episodes'])} games won, "
                              f"{rep['mines']} mines")
        self._draw()
 
    def _advance(self):
        E = self.data["episodes"][self.ep]
        if self.fi == len(E["frames"]) - 1:  # linger on the final board
            if self.held < 2:
                self.held += 1
                return False
            self.held = 0
        if not self._forward():
            if self.pending is not None:  # newer agent ready: switch to it
                self._apply(self.pending)
                return True
            if self.live:  # keep looping while training runs
                self.ep, self.fi = 0, 0
                self._draw()
                return True
            self._set_playing(False)
            return False
        self._draw()
        return True
 
    def _tick(self, _frame):
        changed = False
        with self.shared.lock:
            rep, evals = self.shared.replay, list(self.shared.evals)
            status, finished = self.shared.status, self.shared.finished
        if rep is not None:
            if self.data is None:
                self._apply(rep)
                changed = True
            elif rep["version"] != self.data["version"] and (
                    self.pending is None or self.pending["version"] != rep["version"]):
                self.pending = rep
                self._set_load_button()
                changed = True
        if self.live and len(evals) != self.evals_seen:
            self.evals_seen = len(evals)
            xs = [e[0] for e in evals]
            self.l_win.set_data(xs, [e[1] for e in evals])
            self.l_clr.set_data(xs, [e[2] for e in evals])
            changed = True
        if status != self.last_status:
            self.last_status = status
            self.train_txt.set_text(status)
            changed = True
        if finished and self.b_stop is not None and self.b_stop.label.get_text() != "Training done":
            self.b_stop.label.set_text("Training done")
            self.b_stop.ax.set_facecolor("#e5e7eb")
            self.b_stop.color = "#e5e7eb"
            changed = True
        if self.data is None:
            self.info_txt.set_text("Waiting for the first games...")
        elif self.playing and self._advance():
            changed = True
        if changed:
            self.fig.canvas.draw_idle()
 
    # -------------------------------------------------------------- drawing
    def _draw(self):
        W, H = self.W, self.H
        E = self.data["episodes"][self.ep]
        F = E["frames"][self.fi]
        P = E["frames"][self.fi - 1] if self.fi > 0 else None
        end = self.fi == len(E["frames"]) - 1
        mines = set(E["mines"]) if end else set()
        booms = set(E["detonated"]) if end else set()
        use_q = self.show_q and F["q"] is not None
        lo = hi = 0.0
        nxt = -1
        if use_q:
            vals = [v for v in F["q"] if v is not None]
            if vals:
                lo, hi = min(vals), max(vals)
                nxt = next(i for i, v in enumerate(F["q"]) if v == hi)
 
        for r in range(H):
            for c in range(W):
                i = r * W + c
                v = F["visible"][r][c]
                rect, txt, dot, ring = self.cells[i]
                rect.set_facecolor(self.COV)
                dot.set_visible(False)
                ring.set_visible(False)
                txt.set_text("")
                txt.set_fontsize(self.fs)
                txt.set_fontweight("bold")
                if v >= 0:
                    rect.set_facecolor(self.NEW if (P is not None and P["visible"][r][c] < 0) else self.REV)
                    if v > 0:
                        txt.set_text(str(v))
                        txt.set_color(self.NUM[v])
                elif (r, c) in booms:
                    rect.set_facecolor(self.BOOM)
                    dot.set_visible(True)
                elif F["flags"][r][c]:
                    txt.set_text("F")
                    txt.set_color("#dc2626")
                elif (r, c) in mines:
                    rect.set_facecolor(self.REV)
                    dot.set_visible(True)
                if use_q and F["q"][i] is not None:
                    t = (F["q"][i] - lo) / (hi - lo) if hi > lo else 0.5
                    rect.set_facecolor(self.cmap(0.12 + 0.55 * t))
                    txt.set_text(f"{F['q'][i]:.2f}")
                    txt.set_color("#0f172a")
                    txt.set_fontsize(self.fs * 0.5)
                    txt.set_fontweight("normal")
                    if i == nxt:
                        ring.set_edgecolor("#2563eb")
                        ring.set_linestyle("--")
                        ring.set_visible(True)
                if F["last"] and F["last"][0] == r and F["last"][1] == c:
                    ring.set_edgecolor("#ea580c" if F["last"][2] == "chord" else "#2563eb")
                    ring.set_linestyle("-")
                    ring.set_visible(True)
 
        total = sum(f["reward"] for f in E["frames"][: self.fi + 1])
        act = ("start" if not F["last"] else
               f"{'Chord' if F['last'][2] == 'chord' else 'Reveal'} at row {F['last'][0]}, col {F['last'][1]}")
        self.info_txt.set_text(
            f"Game {self.ep + 1}/{len(self.data['episodes'])}   Move {self.fi}/{len(E['frames']) - 1}\n"
            f"Last action: {act}\n"
            f"Reward: {F['reward']:+.2f}   (total {total:+.2f})\n"
            f"Safe tiles revealed: {F['revealed']} / {self.data['total_safe']}")
        if end:
            if E["won"]:
                self.status_txt.set_text("WON")
                self.status_txt.set_color("#15803d")
            else:
                self.status_txt.set_text("LOST - hit a mine" if E["lost"] else "Stopped (no legal moves)")
                self.status_txt.set_color("#b91c1c")
        else:
            self.status_txt.set_text("Playing")
            self.status_txt.set_color("#555555")
 
 
# ================================================================== training
def train(cfg: Config, viz: SimpleNamespace, shared: SharedState | None = None) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
 
    env = MinesweeperEnv(cfg.width, cfg.height, cfg.num_mines)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
 
    online = MinesweeperDQN(11, cfg.hidden_channels, cfg.num_conv_layers).to(device)
    target = MinesweeperDQN(11, cfg.hidden_channels, cfg.num_conv_layers).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optim = torch.optim.Adam(online.parameters(), lr=cfg.lr)
 
    buffer = ReplayBuffer(cfg.buffer_capacity, obs_shape, n_actions)
    try:
        from torch.utils.tensorboard import SummaryWriter
 
        writer = SummaryWriter(cfg.log_dir)
    except ImportError:
        writer = None
        print("tensorboard not installed - skipping TensorBoard logging")
 
    viz_env = MinesweeperEnv(cfg.width, cfg.height, cfg.num_mines) if shared is not None else None
 
    def push_replay(step: int) -> None:
        """Play a few greedy games on fixed boards and hand them to the window."""
        was_training = online.training
        online.eval()
        label = "Agent during training - " + (f"step {step:,}" if step else "untrained")
        shared.set_replay(build_replay(viz_env, net_policy(online, device), viz.episodes, 0, label, step))
        online.train(was_training)
 
    recent_reward, recent_win, recent_len = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    ep_reward, ep_len = 0.0, 0
    grad_steps, best_win = 0, -1.0
    last_loss = float("nan")
    stopped, step = False, 0
 
    if shared is not None:
        push_replay(0)
        shared.set_status("Training started...")
 
    obs, info = env.reset(seed=cfg.seed)
    for step in range(1, cfg.total_steps + 1):
        if shared is not None and shared.stop.is_set():
            stopped = True
            step -= 1
            break
 
        eps = epsilon_at(step, cfg)
        action = select_action(online, obs, info["action_mask"], eps, device)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, done, info["action_mask"])
        obs = next_obs
        ep_reward += reward
        ep_len += 1
 
        if done:
            recent_reward.append(ep_reward)
            recent_win.append(float(env.won))
            recent_len.append(ep_len)
            ep_reward, ep_len = 0.0, 0
            obs, info = env.reset()
 
        if len(buffer) >= cfg.learning_starts and step % cfg.train_freq == 0:
            batch = buffer.sample(cfg.batch_size, device)
            loss = dqn_loss(online, target, batch, cfg.gamma)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
            optim.step()
            grad_steps += 1
            last_loss = loss.item()
            if grad_steps % cfg.target_update_every == 0:
                target.load_state_dict(online.state_dict())  # theta- <- theta
 
        if step % cfg.log_every == 0:
            if writer is not None:
                writer.add_scalar("train/epsilon", eps, step)
                if not np.isnan(last_loss):
                    writer.add_scalar("train/loss", last_loss, step)
                if recent_win:
                    writer.add_scalar("train/win_rate_100", np.mean(recent_win), step)
                    writer.add_scalar("train/reward_100", np.mean(recent_reward), step)
                    writer.add_scalar("train/ep_len_100", np.mean(recent_len), step)
            if shared is not None:
                w100 = np.mean(recent_win) if recent_win else 0.0
                shared.set_status(f"Training: step {step:,} / {cfg.total_steps:,}  |  "
                                  f"epsilon {eps:.2f}  |  train win rate (last 100) {w100:.2f}")
 
        if shared is not None and step % viz.every == 0:
            push_replay(step)
 
        if step % cfg.eval_every == 0:
            stats = evaluate_policy(online, cfg, cfg.eval_episodes, device=device)
            if writer is not None:
                for k, v in stats.items():
                    writer.add_scalar(f"eval/{k}", v, step)
            if shared is not None:
                shared.add_eval(step, stats["win_rate"], stats["mean_cleared_fraction"])
            print(
                f"step {step:>8d} | eps {eps:.3f} | train win(100) "
                f"{np.mean(recent_win) if recent_win else 0:.3f} | eval win "
                f"{stats['win_rate']:.3f} | eval cleared {stats['mean_cleared_fraction']:.3f}",
                flush=True,
            )
            save_checkpoint(os.path.join(cfg.checkpoint_dir, "latest.pt"), online, cfg, step=step)
            if stats["win_rate"] > best_win:
                best_win = stats["win_rate"]
                save_checkpoint(os.path.join(cfg.checkpoint_dir, "best.pt"), online, cfg, step=step)
 
    if writer is not None:
        writer.close()
    if stopped:
        save_checkpoint(os.path.join(cfg.checkpoint_dir, "latest.pt"), online, cfg, step=step)
    if shared is not None:
        push_replay(step)
        shared.set_status(f"Training {'stopped' if stopped else 'finished'} at step {step:,}. "
                          f"Checkpoints are in '{cfg.checkpoint_dir}/'.")
    print(f"Training {'stopped' if stopped else 'finished'} at step {step:,}.", flush=True)
 
 
def _train_thread(cfg, viz, shared) -> None:
    try:
        train(cfg, viz, shared)
    except Exception:
        traceback.print_exc()
        shared.set_status("Training crashed - see the terminal for the error")
    finally:
        shared.finished = True
 
 
# ============================================================ entry points
def gui_available() -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed (python3 -m pip install matplotlib) - training without the window.")
        return False
    backend = plt.get_backend().lower()
    if backend in {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"} or "inline" in backend:
        print(f"matplotlib backend '{backend}' cannot open windows - training without the window.")
        return False
    return True
 
 
def run_replay(path: str, viz: SimpleNamespace) -> None:
    """Watch a saved agent (no training)."""
    if not gui_available():
        return
    net, ckpt_cfg = load_checkpoint(path, "cpu")
    net.eval()
    env = MinesweeperEnv(ckpt_cfg.width, ckpt_cfg.height, ckpt_cfg.num_mines)
    shared = SharedState()
    shared.set_replay(build_replay(env, net_policy(net, "cpu"), viz.episodes, 0,
                                   f"Saved agent ({os.path.basename(path)})", 1))
    shared.set_status("Replay of a saved checkpoint")
    shared.finished = True
    ReplayWindow(shared, env.width, env.height, live=False).show()
 
 
def run_training_with_window(cfg: Config, viz: SimpleNamespace) -> None:
    shared = SharedState()
    thread = threading.Thread(target=_train_thread, args=(cfg, viz, shared), daemon=True)
    thread.start()
    window = ReplayWindow(shared, cfg.width, cfg.height, total_steps=cfg.total_steps, live=True)
    try:
        window.show()  # blocks until the window is closed
    except KeyboardInterrupt:
        shared.stop.set()
    if thread.is_alive():
        print("Window closed - training continues in the background (Ctrl+C to stop).")
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            shared.stop.set()
            thread.join()
 
 
def parse_args() -> tuple[Config, SimpleNamespace, str | None]:
    """Any Config field can be overridden, e.g. --total_steps 100000 --num_mines 15.
    Window options: --no-viz, --viz_episodes, --viz_every, --replay CHECKPOINT."""
    p = argparse.ArgumentParser()
    for f in fields(Config):
        p.add_argument(f"--{f.name}", type=type(getattr(CONFIG, f.name)), default=None)
    p.add_argument("--no-viz", action="store_true", help="train without the live window")
    p.add_argument("--viz_episodes", type=int, default=5, help="games shown per window update")
    p.add_argument("--viz_every", type=int, default=2000, help="env steps between window updates")
    p.add_argument("--replay", default=None, metavar="CHECKPOINT",
                   help="don't train; just watch a saved agent in the window")
    args = vars(p.parse_args())
    viz = SimpleNamespace(enabled=not args.pop("no_viz"), episodes=args.pop("viz_episodes"),
                          every=args.pop("viz_every"))
    replay_path = args.pop("replay")
    cfg = replace(CONFIG, **{k: v for k, v in args.items() if v is not None})
    return cfg, viz, replay_path
 
 
if __name__ == "__main__":
    cfg, viz, replay_path = parse_args()
    if replay_path:
        run_replay(replay_path, viz)
    elif viz.enabled and gui_available():
        run_training_with_window(cfg, viz)
    else:
        train(cfg, viz, None)