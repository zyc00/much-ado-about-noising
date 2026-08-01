import json, sys
n = 0; ks = set()
for l in open(sys.argv[1]):
    n += 1
    try: ks.update(json.loads(l).keys())
    except: pass
print(n, sorted(ks))
