"""Wrap top-level NAMED default elements in an unnamed <default> for any
kitchen include file that lacks the wrapper (mujoco 3 requirement).
Then compile-test the master model."""
import os
import re

import mujoco

ROOT = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
        "adept_models/kitchen/assets")
for f in sorted(os.listdir(ROOT)):
    if not f.endswith(".xml"):
        continue
    p = os.path.join(ROOT, f)
    s = open(p).read()
    body = re.search(r"<mujoco[^>]*>(.*)</mujoco>", s, re.S)
    if not body:
        continue
    b = body.group(1)
    tops = re.findall(r"<default (class=\"[^\"]+\")>", b)
    # top-level named defaults = named <default> not preceded by an
    # unnamed wrapper: detect files whose FIRST default tag is named
    first = re.search(r"<default[^>]*>", b)
    if first and 'class=' in first.group(0):
        blocks = re.search(r"(<default class=.*</default>)\s*", b, re.S)
        if blocks:
            wrapped = "<default>\n" + blocks.group(1) + "\n</default>"
            s2 = s.replace(blocks.group(1), wrapped, 1)
            open(p, "w").write(s2)
            print("WRAPPED", f)
A = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
     "adept_envs/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml")
try:
    mujoco.MjModel.from_xml_path(A)
    print("COMPILE_OK")
except Exception as e:
    print("COMPILE_FAIL", str(e)[:140])
