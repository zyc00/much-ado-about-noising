"""Wrap kettle_asset.xml's top-level named defaults in an unnamed
<default> (matches sibling files; mujoco 3 requirement). Compile-test."""
import mujoco

K = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
     "adept_models/kitchen/assets/kettle_asset.xml")
s = open(K).read()
if "<default>\n<default class=" in s or "rpl_wrapped" in s:
    print("ALREADY_WRAPPED")
else:
    i = s.find("<default class=")
    j = s.rfind("</default>") + len("</default>")
    assert 0 < i < j, (i, j)
    s2 = s[:i] + "<!--rpl_wrapped--><default>\n" + s[i:j] + "\n</default>" + s[j:]
    open(K, "w").write(s2)
    print("WRAPPED kettle_asset.xml")
A = ("mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning/"
     "adept_envs/adept_envs/franka/assets/franka_kitchen_jntpos_act_ab.xml")
try:
    mujoco.MjModel.from_xml_path(A)
    print("COMPILE_OK")
except Exception as e:
    print("COMPILE_FAIL", str(e)[:140])
