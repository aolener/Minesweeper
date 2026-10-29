# Minesweeper
An engine that plays minesweeper using RL techniques

## System Architecture

This project models Minesweeper as a Markov Decision Process (MDP) designed for deep reinforcement learning algorithms. 

For complete design details, consult the [Architecture Overview](architecture_overview.md), which covers:
* **MDP Formalization:** State representation, reward scaling, and action masking dynamics.
* **Spatial Feature Engine:** 11-channel state tensor configuration `(11 × H × W)` encoding number counts, unrevealed tiles, and flags.
* **Modular Codebase:** Breakdown of environment logic, CNN model architecture, training loops, and evaluation pipelines.

## Setup & Authentication

This repository requires authentication via GitHub. Password authentication is not supported.

### Option 1: HTTPS (Token-based)
If cloning via HTTPS, you must use a **Personal Access Token (PAT)** as your password:
1. Go to **GitHub Settings > Developer Settings > Personal access tokens (classic)**.
2. Generate a token with the `repo` scope enabled.
3. Use your GitHub username and paste the token when your terminal prompts you for a password.

### Option 2: SSH 
To avoid managing tokens, add an SSH key to your GitHub account and clone using the SSH URL:
```bash
git clone git@github.com:aolener/Minesweeper.git
```
