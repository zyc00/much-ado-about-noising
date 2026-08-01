"""Compile every standalone kitchen model; wrap named top-level defaults
in ALL mujocoinclude files; recompile. Prints per-file status."""
import os
import re

import mujoco

ROOT = "mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning"
masters, wrapped = [], 0
for dp, _, fs in os.walk(ROOT):
    for f in fs:
        if not f.endswith(".xml"):
            continue
        p = os.path.join(dp, f)
        s = open(p).read()
        if "<mujocoinclude" in s:
            if "rpl_wrapped" in s:
                continue
            m = re.search(r"<default[^>]*>", s)
            if m and "class=" in m.group(0):
                i = s.find(m.group(0))
                j = s.rfind("</default>") + len("</default>")
                if 0 < i < j:
                    s2 = (s[:i] + "<!--rpl_wrapped--><default>\n" + s[i:j]
                          + "\n</default>" + s[j:])
                    open(p, "w").write(s2)
                    wrapped += 1
                    print("WRAPPED", p[len(ROOT):])
        elif "<mujoco" in s:
            masters.append(p)
print(f"wrapped {wrapped} include files; compiling {len(masters)} masters")
for p in masters:
    try:
        mujoco.MjModel.from_xml_path(p)
        print("OK ", p[len(ROOT):])
    except Exception as e:
        print("FAIL", p[len(ROOT):], str(e)[:70])
