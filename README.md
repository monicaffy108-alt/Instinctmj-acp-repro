# InstinctMJ ACP Reproduction (Unitree G1 · Single RTX 5070)

Reproducing the `acp` branch of [InstinctMJ](https://github.com/project-instinct/InstinctMJ) on a **single 12 GB consumer GPU**.

ACP = Attention-based Contact Planning; the policy is trained with **SuperMimic** (differentiable world model + BPTT).

The upstream branch **does not release its datasets**, so this repo rebuilds the full path from public mocap data to a trainable format, adds single-GPU patches, and documents every pitfall we hit.

---

## 1. Results

Stage-1 (contact-conditioned tracker), 40 LAFAN1-G1 motions, 512 envs, 5000 iterations, ~2 hours.

| Metric | Random policy (0) | 5000 iters |
|---|---|---|
| `sim/episode_length` | 23 steps (0.46 s) | **155.8 steps (3.1 s)** |
| `sim/terminated_fraction` | 0.99 | **0.837** |
| `imagination/survival` (32 steps) | — | **0.795** |
| `imagination/reward` | — | **8.29** |
| `policy/loss` | — | **-13.85** |

![training curves](results/training_curves.png)

Raw scalars: [`results/metrics.csv`](results/metrics.csv).

The policy takes two or three steps in MuJoCo, but `terminated_fraction` is still 0.84 —
it has **not** learned stable walking yet (see §7).

---

## 2. Hardware and software

| Item | Tested configuration |
|---|---|
| GPU | RTX 5070 12 GB (Blackwell, sm_120) |
| OS | Ubuntu, Python 3.12 |
| Key deps | torch 2.10+cu128, mujoco 3.11, mujoco-warp, warp-lang 1.14, mjlab v1.6.0 |
| Env manager | uv |

Measured VRAM: 512 envs with the BPTT policy phase ≈ 9.1 GB / 12 GB.

---

## 3. Quick start

```bash
# 0) Layout: mjlab must be a sibling of InstinctMJ
mkdir -p ~/code && cd ~/code
git clone -b v1.6.0 https://github.com/mujocolab/mjlab.git
git clone -b acp    https://github.com/project-instinct/InstinctMJ.git
cd InstinctMJ && uv sync

# 1) Apply the local patches (camera follow + reset sampling, see §6)
git apply /path/to/instinctmj-acp-repro/patches/0001-local-fixes.patch

# 2) Download public data (Unitree LAFAN1 retargeted to G1, 30 fps CSV)
bash scripts/01_download_data.sh ~/Datasets/LAFAN1_Retargeting

# 3) CSV -> ACP npz (30 -> 50 Hz, quaternions reordered to wxyz)
python scripts/02_csv_to_acp_npz.py \
  ~/Datasets/LAFAN1_Retargeting/g1 \
  ~/Datasets/acp/acp_motions_v1_50hz/flat

# 4) Generate the flat terrain (STL + analytic geometry for SDF contacts)
python scripts/03_make_flat_terrain.py ~/Datasets/acp/acp_motions_v1_50hz

# 5) Build metadata and label contacts (run from the InstinctMJ root)
uv run instinct-motion-metadata --path ~/Datasets/acp/acp_motions_v1_50hz
uv run python -m instinct_mj.scripts.acp.label_contacts \
  --dataset ~/Datasets/acp/acp_motions_v1_50hz --overwrite

# 6) Preflight check -> train -> play
python scripts/04_check_dataset.py ~/Datasets/acp/acp_motions_v1_50hz
./scripts/05_run_training.sh
./scripts/06_play_policy.sh <run_dir_name> model_5000.pt
```

Just want to see it run without training? The repo ships `assets/model_5000.pt`;
copy it into any run directory and play:

```bash
mkdir -p logs/instinct_rl/demo/2026-01-01_00-00-00_demo
cp assets/model_5000.pt logs/instinct_rl/demo/2026-01-01_00-00-00_demo/
./scripts/06_play_policy.sh demo/2026-01-01_00-00-00_demo model_5000.pt
```

TensorBoard (**do not use `uv run`**, see §6.7):

```bash
.venv/bin/tensorboard --logdir logs/instinct_rl --port 6006 --bind_all
```

---

## 4. Building the dataset

### 4.1 Why you have to build it yourself

The `acp` branch README states plainly: **"The datasets are not released with this branch."**
The Google Drive folder linked upstream contains pretrained weights only — no data.
So the dataset must be rebuilt from public mocap sources.

### 4.2 Sources

| Source | Role | Location |
|---|---|---|
| Unitree LAFAN1_Retargeting_Dataset | **Used here**: full-body G1 motions, 30 fps CSV | HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset` |
| BeyondMimic | Alternative (already npz), useful converter reference | UC Berkeley Hybrid Robotics |
| OmniRetarget | Terrain-mesh climbing data (not used here) | HybridRobotics |

### 4.3 Directory layout

```
~/Datasets/acp/acp_motions_v1_50hz/
├── metadata.yaml                     # motion list + terrain mapping
└── flat/                             # one directory per terrain
    ├── flat.stl                      # closed box, top face at z=0
    ├── flat.geoms.yaml               # analytic geometry, must match the STL
    ├── walk1_subject1_retargeted.npz
    ├── walk1_subject1_retargeted.contacts.npz
    └── ...
```

### 4.4 Motion npz format (5 required keys)

| key | shape | Notes |
|---|---|---|
| `framerate` | () | **Must be 50.0** (policy at 50 Hz, PD at 200 Hz) |
| `joint_names` | (J,) | 29 G1 joints; matched by name, order does not matter |
| `joint_pos` | (T, J) | radians |
| `base_pos_w` | (T, 3) | pelvis world position, **metres** (not centimetres) |
| `base_quat_w` | (T, 4) | **wxyz** order (the source CSV is xyzw and must be reordered) |

### 4.5 Contact labels `.contacts.npz`

Produced by `label_contacts`, covering 9 contact bodies
(pelvis, both wrists, elbows, knees, ankles):

| key | shape |
|---|---|
| `framerate` | () |
| `body_names` | (9,) |
| `contact_flag` | (T, 9) bool |
| `contact_pos_w` | (T, 9, 3) float32, zeros where there is no contact |

> If the environment uses `simplified_collision=True` (the v1 default), labels must be
> regenerated with `--simplified-collision` or the collision modes will not match.

---

## 5. Single-GPU tuning

| Parameter | Paper | This repo | Reason |
|---|---|---|---|
| envs | 4096 | 512 (768 works) | 12 GB is not enough |
| `--env.sim.nconmax` | default | **64** | **2.5x speedup**, see §6.2 |
| `max_resample` | 3 | **0** | reset probe allocates 4 GB, see §6.3 |

Throughput with 40 motions:

| envs | time per 100 iters | relative throughput |
|---|---|---|
| 256 | 5 min 00 s | 1.0x |
| 512 | 2 min 22 s | 3.6x |
| 768 | 2 min 00 s | 6.4x |

Counter-intuitively, **fewer envs is not always faster**: with a wrong `nconmax`,
smaller batches end up slower.

---

## 6. Pitfalls we hit (ordered by importance)

### 6.1 Self-collision must stay disabled

Robot collision geoms use `contype=2 / conaffinity=1` and terrain uses `contype=1 / conaffinity=2`,
so robot parts do not collide with each other (2 & 1 = 0) while every robot geom still
collides with the terrain. Enabling self-collision adds 906 constraint pairs and makes
training stall or diverge. This is the default in the upstream `assets/acp_g1.py` — **do not change it**.

### 6.2 The default `nconmax` is far too large

The per-env contact buffer is oversized: it wastes VRAM and lets contact detection spin on
empty work. Setting it to 64:

- brings 512 envs down to 4.9 GB (the default OOMs outright)
- cuts 100 iterations from 5:00 to 2:22

```bash
--env.sim.nconmax 64
```

### 6.3 A 4 GB temporary probe on reset

The reset term builds a private model to probe for ground penetration; with 40 motions it
requests 4 GB and OOMs immediately. Setting `max_resample` to 0
(`patches/0001-local-fixes.patch`) avoids it — legal-start filtering already prevents
penetrating start frames.

### 6.4 The policy metrics are NaN for the first 500 iterations — this is expected

`wm_only_iters = 500`: the first 500 iterations **train the world model only**.
Policy metrics (`pi/v/imag_r/surv`) average over an empty buffer and deliberately return
`nan`. Do not read this as divergence. Policy training starts at iteration 500.

### 6.5 VRAM grows during the policy phase

The world-model phase peaks at 5.4 GB; entering the policy phase (32-step BPTT stores
intermediates) pushes it to 9.1 GB. At that point you **cannot** open a playback viewer,
or you get `Warp CUDA error 2: out of memory`.

### 6.6 The playback window shows no robot

`G1AcpTrackV1EnvCfg_PLAY` defaults to a camera with `origin_type=WORLD`, pinned to the world
origin; the robot spawns elsewhere and walks out of frame. Switching to `ASSET_BODY` with
`body_name="torso_link"` makes the camera follow.

### 6.7 Never run TensorBoard through `uv run`

`uv run` restores the environment from `uv.lock`, which pins setuptools to 82 — and that
release **removed `pkg_resources`**, so TensorBoard 2.20 dies with
`ModuleNotFoundError`. Invoke `.venv/bin/tensorboard` directly.

### 6.8 Resume semantics

- `--agent.load-run` expects the **run subdirectory name**
  (e.g. `2026-11-15_21-41-47_g1AcpTrackV1_supermimic`), not a full path;
  pair it with `--agent.experiment-name` to select the experiment directory
- `--checkpoint-file` must be an **absolute path**
- Resuming needs both `model_*.pt` and `acp_state_*.pt` (the latter holds the world model,
  critic and optimizer state)

### 6.9 After changing reward / termination / observation

You must mirror the change in `src/instinct_mj/rl/acp/mdp.py`
(`stage3_mdp.py` for Stage 3), otherwise the environment and the imagination MDP diverge
and the imagined rollouts become meaningless.

---

## 7. Known limitations

1. **Substitute data**: the official ACP corpus (13.2 minutes, with stairs and climbing
   terrain) is unreleased. This repo uses flat-ground LAFAN1 motions instead, so it
   **reproduces Stage-1 flat-ground tracking only** — not the paper's terrain traversal
2. **1/8 of the environments**: 4096 → 512, an order of magnitude less sample throughput,
   so more iterations are needed to converge
3. **Not converged**: at 5000 iterations `terminated_fraction` is still 0.84 and
   `track_contact_label` only reaches 0.46/5.0 — foot-contact timing is not learned;
   expect to need 15k+ iterations
4. **sm_120 friction**: warp occasionally reports `CUDA error 100` (child processes cannot
   see the GPU); it does not affect the main run
5. **Suspicious reward weight**: `dof_pos_limits = -10.0` is high and may suppress stride
   length; no ablation has been run yet

---

## 8. Repository layout

```
├── README.md
├── patches/0001-local-fixes.patch   # 2 upstream edits for single-GPU use
├── scripts/
│   ├── 01_download_data.sh          # download Unitree LAFAN1-G1 CSV
│   ├── 02_csv_to_acp_npz.py         # 30→50 Hz, quaternion xyzw→wxyz
│   ├── 03_make_flat_terrain.py      # flat STL + SDF geometry
│   ├── 04_check_dataset.py          # 7-class preflight check
│   ├── 05_run_training.sh           # single-GPU training/resume wrapper
│   ├── 06_play_policy.sh            # play a policy (camera follows)
│   ├── play_reference.py            # replay reference motion (bypasses the policy)
│   └── live_view.sh                 # reload the latest checkpoint while training
├── assets/
│   ├── model_5000.pt                # trained Stage-1 policy (play it directly)
│   ├── flat.stl                     # 29x18x0.5 m closed box
│   ├── flat.geoms.yaml              # analytic box geometry
│   └── metadata.example.yaml        # metadata sample for 40 motions
└── results/
    ├── metrics.csv                  # all scalars, 450 sampled points
    └── training_curves.png
```

> The dataset (241 MB) and training logs (4.1 GB) are intentionally not committed:
> the data is fully reproducible from public sources via the scripts, and the logs are best
> viewed in TensorBoard — their scalars are exported to `results/metrics.csv`.

---

## 9. Citation and credits

```bibtex
@software{instinctmj_acp,
  title  = {InstinctMJ (acp branch)},
  author = {Project Instinct},
  url    = {https://github.com/project-instinct/InstinctMJ}
}
```

The original mocap comes from Ubisoft La Forge (LAFAN1); the G1 retargeted release is
published by Unitree. This repository contains reproduction scripts and configurations only
and redistributes none of the original data files.
