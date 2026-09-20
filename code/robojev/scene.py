"""Display-only MuJoCo kitchen. The agent never reads these pixels."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from robojev.types import Skill
from robojev.world import World

SCENE = """
<mujoco model="robojev_kitchen">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.55 0.55 0.52" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.94 0.92 0.88" rgb2="0.86 0.83 0.78"/>
    <material name="floor" texture="floor" texrepeat="7 7" reflectance="0.04"/>
    <material name="wood" rgba="0.58 0.39 0.22 1"/>
    <material name="apple" rgba="0.75 0.12 0.1 1"/>
    <material name="leaf" rgba="0.25 0.48 0.22 1"/>
    <material name="shell" rgba="0.94 0.95 0.96 1"/>
    <material name="door" rgba="0.80 0.84 0.88 1"/>
    <material name="cavity" rgba="0.55 0.60 0.64 1"/>
    <material name="robot" rgba="0.16 0.31 0.58 1"/>
  </asset>
  <worldbody>
    <light pos="0.1 -0.2 2.6" dir="0 0.15 -1" diffuse="1 0.98 0.95"/>
    <geom type="plane" size="3 3 0.1" material="floor"/>
    <geom type="box" pos="0 0 0.36" size="0.78 0.42 0.035" material="wood"/>
    <geom type="cylinder" pos="-0.62 0.30 0.18" size="0.035 0.18" material="wood"/>
    <geom type="cylinder" pos="0.62 0.30 0.18" size="0.035 0.18" material="wood"/>
    <geom type="cylinder" pos="-0.62 -0.30 0.18" size="0.035 0.18" material="wood"/>
    <geom type="cylinder" pos="0.62 -0.30 0.18" size="0.035 0.18" material="wood"/>

    <body name="fridge" pos="0.40 0.02 0.74">
      <geom type="box" pos="0 0.02 0" size="0.18 0.15 0.28" material="cavity"/>
      <geom type="box" pos="0 0.16 0" size="0.20 0.02 0.30" material="shell"/>
      <geom type="box" pos="0 0 0.30" size="0.20 0.17 0.02" material="shell"/>
      <geom type="box" pos="0 0 -0.30" size="0.20 0.17 0.02" material="shell"/>
      <geom type="box" pos="0.20 0 0" size="0.02 0.17 0.30" material="shell"/>
      <body name="door" pos="-0.20 -0.15 0">
        <joint name="door" type="hinge" axis="0 0 1" range="-1.7 0.05" limited="true"/>
        <geom type="box" pos="0.20 -0.015 0" size="0.20 0.016 0.30" material="door"/>
        <geom type="capsule" fromto="0.32 -0.04 -0.08 0.32 -0.04 0.08" size="0.012" rgba="0.32 0.35 0.4 1"/>
      </body>
    </body>

    <body name="focus" pos="0.05 0 0.7"/>
    <body name="apple" mocap="true" pos="-0.32 0 0.44">
      <geom type="sphere" size="0.045" material="apple"/>
      <geom type="capsule" fromto="0 0 0.03 0.01 0 0.055" size="0.008" material="leaf"/>
    </body>
    <body name="gripper" mocap="true" pos="-0.32 -0.22 0.68">
      <geom type="capsule" fromto="0 0 0.02 0 0 0.18" size="0.028" material="robot"/>
      <geom type="box" pos="0.04 0 0" size="0.012 0.018 0.028" material="robot"/>
      <geom type="box" pos="-0.04 0 0" size="0.012 0.018 0.028" material="robot"/>
    </body>
    <camera name="show" mode="targetbody" target="focus" pos="0.15 -1.75 1.28" fovy="46"/>
  </worldbody>
</mujoco>
"""

APPLE_COUNTER = np.array([-0.32, 0.0, 0.44])
APPLE_INSIDE = np.array([0.38, -0.10, 0.70])
GRIP_READY = np.array([-0.32, -0.18, 0.66])
GRIP_MISS = np.array([-0.18, 0.08, 0.50])
GRIP_CARRY = np.array([-0.02, -0.22, 0.78])
GRIP_AT_FRIDGE = np.array([0.08, -0.42, 0.72])
DOOR_OPEN = -1.2


class KitchenRenderer:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.apple_id = int(self.model.body_mocapid[self._body("apple")])
        self.grip_id = int(self.model.body_mocapid[self._body("gripper")])
        self.door_adr = int(self.model.jnt_qposadr[self._joint("door")])

    def snap(self, world: World, skill: Skill | None, note: str, path: Path) -> None:
        grip, apple, door = _pose(world, skill, note)
        self.data.mocap_pos[self.grip_id] = grip
        self.data.mocap_pos[self.apple_id] = apple
        self.data.qpos[self.door_adr] = door
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="show")
        pixels = self.renderer.render()
        Image.fromarray(pixels).save(path)
        print(f"frame {path.name} max={int(pixels.max())} mean={float(pixels.mean()):.1f}", flush=True)
        if int(pixels.max()) == 0:
            raise RuntimeError(f"black frame: {path}")

    def _body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _joint(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _pose(world: World, skill: Skill | None, note: str) -> tuple[np.ndarray, np.ndarray, float]:
    holding = world.held == "apple"
    door = DOOR_OPEN if world.open.get("fridge") else 0.0
    slipped = "slipped" in note
    if slipped:
        grip = GRIP_MISS.copy()
    elif holding and world.at == "fridge":
        grip = GRIP_AT_FRIDGE.copy()
    elif holding:
        grip = GRIP_CARRY.copy()
    elif world.at == "fridge":
        grip = GRIP_AT_FRIDGE.copy()
    else:
        grip = GRIP_READY.copy()

    if holding:
        apple = grip + np.array([0.0, 0.04, -0.09])
    elif world.loc.get("apple") == "fridge":
        apple = APPLE_INSIDE.copy()
    else:
        apple = APPLE_COUNTER.copy()
    if skill and skill.name == "close" and not world.open.get("fridge"):
        grip = GRIP_AT_FRIDGE.copy()
    return grip, apple, door
