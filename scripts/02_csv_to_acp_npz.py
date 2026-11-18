"""把 Unitree LAFAN1_Retargeting_Dataset 的 G1 CSV 转成 ACP (InstinctMJ) 能读的 npz。

CSV 格式(官方数据集卡片):
    36 列 = x y z | qx qy qz qw | 29 个关节角, 30 FPS, 无表头

ACP 需要的 npz key:
    framerate   标量 float, 50.0
    joint_names 29 个关节名字符串
    joint_pos   (T, 29) float, 弧度
    base_pos_w  (T, 3)  float, 米
    base_quat_w (T, 4)  float, 四元数 **wxyz** 顺序(CSV 里是 xyzw, 必须重排)

用法:
    python csv_to_acp_npz.py <csv文件或目录> <输出目录>
"""

from __future__ import annotations

import os
import sys

import numpy as np

SRC_FPS = 30.0
DST_FPS = 50.0

# Unitree G1 的 29 关节顺序(与 CSV 列序一致)
JOINT_NAMES = [
    # 左腿
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # 右腿
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # 右臂
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """四元数球面插值, q 为 (..., 4)。alpha 为同长度的一维权重。"""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    # 半球对齐: 让两帧四元数在同一半球, 避免走远路
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = alpha[..., None]
    # sin_theta 接近 0 时退化为线性插值
    w0 = np.where(sin_theta > 1e-6, np.sin((1.0 - a) * theta) / sin_theta, 1.0 - a)
    w1 = np.where(sin_theta > 1e-6, np.sin(a * theta) / sin_theta, a)
    return w0 * q0 + w1 * q1


def resample(raw: np.ndarray, dst_fps: float, src_fps: float) -> np.ndarray:
    """把 (T, C) 的序列从 src_fps 重采样到 dst_fps。位置/关节用线性, 四元数用 slerp。"""
    n_src = raw.shape[0]
    pos = raw[:, 0:3]
    quat_xyzw = raw[:, 3:7]
    joints = raw[:, 7:]

    t_src = np.arange(n_src) / src_fps
    duration = t_src[-1]
    n_dst = int(np.floor(duration * dst_fps)) + 1
    t_dst = np.arange(n_dst) / dst_fps

    # 找到每个目标时刻左右两帧
    src_idx = t_dst * src_fps
    i0 = np.clip(np.floor(src_idx).astype(np.int64), 0, n_src - 1)
    i1 = np.clip(i0 + 1, 0, n_src - 1)
    alpha = src_idx - i0

    # 位置 / 关节: 逐列线性插值
    pos_dst = pos[i0] + (pos[i1] - pos[i0]) * alpha[:, None]
    joints_dst = joints[i0] + (joints[i1] - joints[i0]) * alpha[:, None]

    # 四元数: 转 wxyz 后 slerp
    quat_wxyz_src = np.stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]], axis=1
    )
    quat_dst = slerp(quat_wxyz_src[i0], quat_wxyz_src[i1], alpha)
    # 归一化, 消除插值带来的数值漂移
    quat_dst /= np.linalg.norm(quat_dst, axis=1, keepdims=True)

    return np.concatenate([pos_dst, quat_dst, joints_dst], axis=1)


def convert_one(csv_path: str, out_dir: str) -> dict:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != 36:
        raise ValueError(f"{csv_path}: 期望 36 列, 实际 {raw.shape[1]} 列")

    out = resample(raw, DST_FPS, SRC_FPS)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_retargeted.npz")
    np.savez(
        out_path,
        framerate=np.array(DST_FPS),
        joint_names=np.array(JOINT_NAMES),
        joint_pos=out[:, 7:].astype(np.float32),
        base_pos_w=out[:, 0:3].astype(np.float32),
        base_quat_w=out[:, 3:7].astype(np.float32),
    )
    return {"name": stem, "n_src": raw.shape[0], "n_dst": out.shape[0], "path": out_path}


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    csvs = (
        [src]
        if os.path.isfile(src)
        else sorted(
            os.path.join(src, f) for f in os.listdir(src) if f.endswith(".csv")
        )
    )
    for csv_path in csvs:
        info = convert_one(csv_path, out_dir)
        print(
            f"{info['name']:28s} {info['n_src']:6d} 帧(30fps) -> {info['n_dst']:6d} 帧(50fps)"
            f"  比值={info['n_dst'] / info['n_src']:.3f}"
        )
    print(f"\n完成: {len(csvs)} 个文件 -> {out_dir}")


if __name__ == "__main__":
    main()
