"""直接在 MuJoCo 查看器里播放参考运动数据（不经过策略，纯回放）。

用法:
    uv run python play_reference.py <npz文件路径>

例如播放走路动作:
    uv run python play_reference.py ~/Datasets/acp/acp_motions_v1_50hz/flat/walk1_subject1_retargeted.npz

操作:
    空格: 暂停/继续
    鼠标拖拽: 旋转视角
    滚轮: 缩放
"""
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

# G1 模型路径（ACP 用的版本）
MODEL_XML = "/home/monicaffy/InstinctMJ/src/instinct_mj/assets/resources/acp_g1/g1_29dof.xml"
# 地形 STL
TERRAIN_STL = "/home/monicaffy/Datasets/acp/acp_motions_v1_50hz/flat/flat.stl"


def main():
    if len(sys.argv) not in (2, 3):
        print("用法: python play_reference.py <npz文件> [地形stl文件]")
        sys.exit(1)

    npz_path = sys.argv[1]
    terrain_stl = sys.argv[2] if len(sys.argv) == 3 else TERRAIN_STL
    data = np.load(npz_path, allow_pickle=True)
    joint_names = list(data["joint_names"])
    joint_pos = data["joint_pos"]  # (T, 29)
    base_pos = data["base_pos_w"]  # (T, 3)
    base_quat = data["base_quat_w"]  # (T, 4) wxyz
    fps = float(data["framerate"])
    n_frames = joint_pos.shape[0]

    print(f"帧数: {n_frames}, 帧率: {fps} Hz, 时长: {n_frames / fps:.1f} 秒")

    # 加载机器人模型 XML, 在 <worldbody> 里插入地形 mesh + geom
    with open(MODEL_XML, "r") as f:
        xml = f.read()

    terrain_basename = os.path.basename(terrain_stl)

    # <mesh> 放在 <asset> 里, <geom> 放在 <worldbody> 里
    mesh_def = f'    <mesh name="terrain_flat" file="{terrain_basename}"/>\n'
    geom_def = f'''    <geom name="terrain_geom" type="mesh" mesh="terrain_flat"
          contype="1" conaffinity="2" rgba="0.5 0.5 0.5 1"/>
'''
    xml = xml.replace("</asset>", mesh_def + "</asset>")
    xml = xml.replace("</worldbody>", geom_def + "</worldbody>")

    # 写临时 XML 到模型目录, 让 meshdir="meshes" 能找到 flat.stl
    model_dir = os.path.dirname(MODEL_XML)
    tmp_xml = os.path.join(model_dir, "_g1_with_terrain.xml")
    with open(tmp_xml, "w") as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_path(tmp_xml)
    sim = mujoco.MjData(model)

    # 建立 npz 关节名 -> 模型关节名的映射
    model_joint_names = [model.joint(i).name for i in range(model.njnt)]
    # 前 7 个是 freejoint (浮动基座), 后面是 actuated joints
    # ACP 模型: 第 0 个是 floating_base_joint, 后面是 23 个驱动关节
    actuated_names = [n for n in model_joint_names if n != "floating_base_joint"]

    # 建立映射: 对于模型里的每个驱动关节, 找到它在 npz 里的列索引
    qpos_idx = []  # 模型 qpos 中的列索引
    data_col = []  # npz joint_pos 中的列索引
    for mj_name in actuated_names:
        if mj_name in joint_names:
            # 找到这个关节在模型 qpos 里的位置
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            qadr = model.jnt_qposadr[jid]
            qpos_idx.append(qadr)
            data_col.append(joint_names.index(mj_name))

    print(f"匹配到 {len(qpos_idx)}/{len(actuated_names)} 个关节")

    frame = 0
    paused = False
    speed = 1.0  # 播放速度倍率

    # 找到 pelvis body id, 用于相机追踪
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    with mujoco.viewer.launch_passive(model, sim) as viewer:
        # 设置相机追踪模式, 让相机跟随机器人
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 3.0   # 初始距离
        viewer.cam.elevation = -20  # 俯角

        while viewer.is_running():
            if not paused:
                frame = (frame + int(speed)) % n_frames

            # 写入基座位置和朝向
            sim.qpos[0:3] = base_pos[frame]
            sim.qpos[3:7] = base_quat[frame]  # wxyz, mujoco 也是 wxyz
            # 写入各关节角
            for qi, ci in zip(qpos_idx, data_col):
                sim.qpos[qi] = joint_pos[frame, ci]

            mujoco.mj_forward(model, sim)

            # 相机跟随: 把观察点设到 pelvis 位置
            viewer.cam.lookat[:] = sim.xpos[pelvis_id]

            viewer.sync()

            # 控制播放速度
            time.sleep(1.0 / (fps * speed))


if __name__ == "__main__":
    main()
