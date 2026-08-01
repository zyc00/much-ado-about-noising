"""Rename 'main' default class in vendored relay-kitchen MJCF (mujoco 3)."""
import os
import re

ROOT = "mip/envs/kitchen/kitchen_thirdparty/relay_policy_learning"
found = patched = 0
for dp, _, fs in os.walk(ROOT):
    for f in fs:
        if not f.endswith(".xml"):
            continue
        p = os.path.join(dp, f)
        s = open(p).read()
        if 'class="main"' in s or "class='main'" in s:
            found += 1
            s2 = (s.replace('class="main"', 'class="rpl_main"')
                    .replace("class='main'", "class='rpl_main'"))
            open(p, "w").write(s2)
            patched += 1
print(f"XMLFIX found={found} patched={patched}")
chk = sum(1 for dp, _, fs in os.walk(ROOT) for f in fs
          if f.endswith(".xml") and "rpl_main" in open(os.path.join(dp, f)).read())
print(f"XMLFIX verify rpl_main in {chk} files")
