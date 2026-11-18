#!/usr/bin/env python3
"""生成 ACP 需要的平地地形：flat.stl + flat.geoms.yaml

要点：
- STL 必须是一个**封闭的立方体**（watertight），顶面在 z=0，机器人在 z=0 上行走
- geoms.yaml 是给 SDF 接触检测用的解析几何，必须和 STL 完全一致
- 尺寸按最大动作位移留足余量（这里 29 x 18 m）

用法:
    python 03_make_flat_terrain.py <数据集根目录>
    例如 python 03_make_flat_terrain.py ~/Datasets/acp/acp_motions_v1_50hz
"""

from __future__ import annotations

import os
import sys

# 半长 (x, y)，厚度
HALF_X, HALF_Y, THICK = 14.5, 9.0, 0.25

TRIS = [
    # 顶面 z=0
    ((+1, +1, 0), (-1, +1, 0), (-1, -1, 0)),
    ((+1, +1, 0), (-1, -1, 0), (+1, -1, 0)),
    # 底面
    ((+1, +1, -2), (-1, -1, -2), (-1, +1, -2)),
    ((+1, +1, -2), (+1, -1, -2), (-1, -1, -2)),
    # 四个侧面
    ((+1, +1, 0), (-1, +1, 0), (-1, +1, -2)),
    ((+1, +1, 0), (-1, +1, -2), (+1, +1, -2)),
    ((-1, +1, 0), (-1, -1, 0), (-1, -1, -2)),
    ((-1, +1, 0), (-1, -1, -2), (-1, +1, -2)),
    ((-1, -1, 0), (+1, -1, 0), (+1, -1, -2)),
    ((-1, -1, 0), (+1, -1, -2), (-1, -1, -2)),
    ((+1, -1, 0), (+1, +1, 0), (+1, +1, -2)),
    ((+1, -1, 0), (+1, +1, -2), (+1, -1, -2)),
]


def scale(p):
    x, y, z = p
    return (
        x * HALF_X,
        y * HALF_Y,
        0.0 if z == 0 else -2 * THICK,
    )


def write_stl(path: str) -> None:
    with open(path, "w") as f:
        f.write("solid flat\n")
        for tri in TRIS:
            f.write("  facet normal 0 0 0\n    outer loop\n")
            for v in tri:
                x, y, z = scale(v)
                f.write(f"      vertex {x:.6f} {y:.6f} {z:.6f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid flat\n")


def write_geoms(path: str) -> None:
    with open(path, "w") as f:
        f.write("- type: box\n")
        f.write(f"  pos: [0.0, 0.0, {-THICK}]\n")
        f.write("  quat: [1.0, 0.0, 0.0, 0.0]\n")
        f.write(f"  size: [{HALF_X}, {HALF_Y}, {THICK}]\n")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    flat_dir = os.path.join(sys.argv[1], "flat")
    os.makedirs(flat_dir, exist_ok=True)
    stl = os.path.join(flat_dir, "flat.stl")
    geoms = os.path.join(flat_dir, "flat.geoms.yaml")
    write_stl(stl)
    write_geoms(geoms)
    print(f"写出 {stl} ({os.path.getsize(stl)} B)")
    print(f"写出 {geoms} ({os.path.getsize(geoms)} B)")


if __name__ == "__main__":
    main()
