"""Show mujoco version and every <default> declaration in the kitchen
include tree, pod-side."""
import os
import re

import mujoco

print("mujoco", mujoco.__version__)
A = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
     "adept_envs/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml")
seen = set()


def scan(p, depth=0):
    p = os.path.normpath(p)
    if p in seen or not os.path.exists(p):
        print("  " * depth + f"[missing/dup] {p}" if p not in seen else "", end="")
        return
    seen.add(p)
    s = open(p).read()
    defs = re.findall(r"<default[^>]*>", s)[:3]
    print("  " * depth + os.path.basename(p), "DEFAULTS:", defs)
    for inc in re.findall(r'<include\s+file="([^"]+)"', s):
        scan(os.path.join(os.path.dirname(p), inc), depth + 1)


scan(A)
try:
    mujoco.MjModel.from_xml_path(A)
    print("COMPILE_OK")
except Exception as e:
    print("COMPILE_FAIL", str(e)[:120])
