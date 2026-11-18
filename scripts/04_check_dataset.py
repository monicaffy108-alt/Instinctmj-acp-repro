#!/usr/bin/env python3
"""训练前自检：验证 ACP 数据集格式正确，避免跑几个小时后才发现数据有问题。

检查项：
1. metadata.yaml / 地形 STL / geoms.yaml 是否存在
2. 每条动作的 5 个必需 key 是否齐全、shape 是否正确
3. framerate 是否为 50
4. 四元数是否已归一化（ACP 必须是 wxyz 且模长 ≈ 1）
5. 关节名是否覆盖 ACP 的 23 个驱动关节
6. 骨盆高度是否合理（G1 站姿约 0.6-0.9 m）
7. 接触标签 sidecar 是否齐全且帧数匹配

用法:
    python 04_check_dataset.py ~/Datasets/acp/acp_motions_v1_50hz
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

ACP_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
REQUIRED_KEYS = ["framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"]


def check_one(path: str) -> list[str]:
    errs: list[str] = []
    d = np.load(path, allow_pickle=False)

    missing = [k for k in REQUIRED_KEYS if k not in d.files]
    if missing:
        return [f"缺少 key: {missing}"]

    fps = float(d["framerate"])
    if abs(fps - 50.0) > 1e-6:
        errs.append(f"framerate={fps}，ACP 要求 50")

    jp, bp, bq = d["joint_pos"], d["base_pos_w"], d["base_quat_w"]
    if not (jp.shape[0] == bp.shape[0] == bq.shape[0]):
        errs.append(f"帧数不一致: joint {jp.shape[0]} / pos {bp.shape[0]} / quat {bq.shape[0]}")
    if bp.shape[1] != 3:
        errs.append(f"base_pos_w 应为 (T,3)，实际 {bp.shape}")
    if bq.shape[1] != 4:
        errs.append(f"base_quat_w 应为 (T,4)，实际 {bq.shape}")

    norms = np.linalg.norm(bq, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        errs.append(f"四元数未归一化: min={norms.min():.4f} max={norms.max():.4f}")

    for name, arr in (("joint_pos", jp), ("base_pos_w", bp), ("base_quat_w", bq)):
        if not np.isfinite(arr).all():
            errs.append(f"{name} 含 NaN/Inf")

    names = [str(x) for x in d["joint_names"]]
    lack = [j for j in ACP_JOINTS if j not in names]
    if lack:
        errs.append(f"缺少 ACP 关节: {lack}")

    # 骨盆高度：倒地/起身类动作最低可达 ~0.1 m，属正常；只有穿地或高得离谱才是错误
    z = bp[:, 2]
    if z.min() < -0.05:
        errs.append(f"骨盆 z 最低 {z.min():.2f} m，疑似穿地")
    if z.max() > 1.2:
        errs.append(f"骨盆 z 最高 {z.max():.2f} m，超出 G1 合理站高")

    # 接触标签 sidecar
    cpath = path.replace(".npz", ".contacts.npz")
    if not os.path.exists(cpath):
        errs.append(f"缺少接触标签 {os.path.basename(cpath)}（跑 label_contacts）")
    else:
        c = np.load(cpath, allow_pickle=False)
        for k in ("framerate", "body_names", "contact_flag", "contact_pos_w"):
            if k not in c.files:
                errs.append(f"接触标签缺少 key {k}")
                break
        else:
            if c["contact_flag"].shape[0] != jp.shape[0]:
                errs.append(f"接触标签帧数 {c['contact_flag'].shape[0]} != 动作 {jp.shape[0]}")
    return errs


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]

    meta_path = os.path.join(root, "metadata.yaml")
    if not os.path.exists(meta_path):
        print(f"[FAIL] 缺少 {meta_path}（用 instinct-motion-metadata 生成）")
        sys.exit(1)
    meta = yaml.safe_load(open(meta_path))
    motions = meta.get("motion_files", [])
    terrains = meta.get("terrains", [])
    print(f"metadata: {len(motions)} 条动作, {len(terrains)} 个地形")

    for t in terrains:
        stl = os.path.join(root, t["terrain_file"])
        geoms = stl.replace(".stl", ".geoms.yaml")
        print(f"  地形 {t.get('terrain_id')}: STL {'OK' if os.path.exists(stl) else '缺失'}"
              f" / geoms {'OK' if os.path.exists(geoms) else '缺失(将退回 force_head)'}")

    bad = 0
    for m in motions:
        rel = m["motion_file"]
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"[FAIL] {rel}: 文件不存在")
            bad += 1
            continue
        errs = check_one(p)
        if errs:
            bad += 1
            print(f"[FAIL] {rel}")
            for e in errs:
                print(f"       - {e}")

    total = len(motions)
    print(f"\n结果: {total - bad}/{total} 条通过")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
