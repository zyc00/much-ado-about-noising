import h5py
src = h5py.File('data/tool_hang_human_lowdim_up.hdf5', 'r')
dst = h5py.File('data/tool_hang_distmlp_clean.hdf5', 'a')
for k, v in src['data'].attrs.items():
    dst['data'].attrs[k] = v
for k, v in src.attrs.items():
    dst.attrs[k] = v
s0 = src['data/demo_0']
for dk in dst['data']:
    for ak, av in s0.attrs.items():
        if ak not in dst['data'][dk].attrs:
            dst['data'][dk].attrs[ak] = av
dst.close(); src.close()
print('ATTRS copied')
