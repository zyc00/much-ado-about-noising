"""Bisect which include file trips the mujoco-3 default error."""
import os
import re

import mujoco

A = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
     "adept_envs/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml")
s = open(A).read()
incs = re.findall(r'<include\s+file="[^"]+"\s*/>', s)
print(f"{len(incs)} includes")


def try_compile(txt, tag):
    tmp = A + ".bisect.xml"
    open(tmp, "w").write(txt)
    try:
        mujoco.MjModel.from_xml_path(tmp)
        print(f"OK_WITHOUT {tag}")
    except Exception as e:
        msg = str(e)[:60]
        print(f"FAIL_WITHOUT {tag}: {msg}")
    finally:
        os.remove(tmp)


for inc in incs:
    try_compile(s.replace(inc, "", 1), re.search(r'file="([^"]+)"', inc).group(1).split("/")[-1])

K = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
     "adept_models/kitchen/assets/kettle_asset.xml")
ks = open(K).read()
m = re.search(r"<default.*?</default>\s*</default>|<default.*?</default>", ks, re.S)
print("KETTLE_DEFAULT_SECTION >>>")
print(ks[:600])
