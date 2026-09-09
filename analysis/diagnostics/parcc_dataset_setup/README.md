# PARCC dataset setup — 2026-09-08

## STOPPED at user request

The user requested stopping the current transfers. Both local systemd units
were stopped: parcc-datasets-20260908 and parcc-data-portforward-20260908.
The task-only source rsync daemon on port 28731 was also stopped after checking
its PID/command. No automatic restart or replacement download was launched.
The user's original SSH control session/reverse tunnel was left running.
Downloaded/staged files, PARCC partial datasets, and logs were preserved.
The per-dataset status JSON files below record the last phase BEFORE stopping;
they are historical and must not be read as evidence of a running transfer.

Target root explicitly supplied by user:
`/vast/projects/jiayuanm/mao-lab/yuchen/`.
Existing `code/` and `data/` directories were empty at inspection.
No training jobs are launched by this setup; code directory is unchanged.

| Target under data/ | Exact existing training source on PFS | Scope |
|---|---|---|
| widowx/ | /mnt/pfs/yuchen/groot/bridge_orig_lerobot/ | Bridge/WidowX, 53,192 episodes |
| fractal/ | /mnt/pfs/yuchen/groot/fractal_lerobot/ | Google Robot, 87,212 episodes |
| gr1/ | /mnt/pfs/yuchen/groot/lerobot/LeRobot/ | 24 gr1_unified task directories |

These are the converted LeRobot datasets used by existing GR00T experiments,
not a fresh RLDS conversion. In particular GR1 is NOT lerobot_kitchen (Panda).
Keep meta/ (including normalization/modality definitions), parquet, and videos.
Download caches `.cache`, `._____temp`, `.msc`, `.mv` are excluded.
GR1 `.jpgpack` decoded-frame caches are also excluded: the original 24,000
MP4 videos are retained and the cache can be rebuilt locally if desired.
Source symlinks are dereferenced so the destination is self-contained.

## Transfer

Persistent local coordinator:
`scripts/cluster/setup_parcc_datasets.py`.
Source inventory builder: `scripts/cluster/parcc_dataset_inventory.py`.
Public-file accelerator: `scripts/cluster/parcc_public_seed.py`.
Initial kubectl-exec rsync adapter `scripts/cluster/parcc_source_rsh.sh` is
retained for reference, but replaced for this transfer due to low throughput
and intermittent exec-stream disconnects.

Uses existing SSH control socket `/tmp/parcc-jigu-control` over the user's
reverse tunnel, connecting to PARCC login01 as zyc0187. No credentials or SSH
private keys are copied to either cluster, no additional public service exposed.
The tunnel/control session must remain available while transferring.
User supplied forwarded SSH agent at `/tmp/ssh-XXXXPnc4La/agent.658786`.
Agent identity is readable, but a fresh connection to localhost:22222 as zyc0187
only offers gssapi-with-mic/keyboard-interactive, so public-key authentication
does not replace the established login session. Existing control session works.
The endpoint ED25519 host key was matched against /etc/ssh public host keys over
the established session and stored in known_hosts; host checks were not disabled.

PFS metadata -> local resumable staging -> PARCC, three dataset workers.
Source read-only rsync daemon binds ONLY pod localhost:28731; a local-only
kubectl port-forward exposes it on workstation localhost:28731. No NodePort,
public endpoint, or write module was created. Modules only expose these three
datasets plus generated source inventories. Daemon config is
scripts/cluster/parcc_rsyncd.conf. Port-forward is a user systemd service:
parcc-data-portforward-20260908.service.

The cross-network source path measured only about 43 KB/s on a 7.4 MB probe.
To avoid moving all payload over that path, public original data/video assets
seed the local staging directory before exact PFS synchronization. Only source
inventory paths with matching byte size are installed, via atomic rename.
For GR1, existing local matching-size files are also copied as seed (not trusted
as final data). Final PFS checksum synchronization is authoritative and repairs
any content differences; public originals do NOT replace our training metadata.
Public seed revisions:
- IPEC-COMMUNITY/bridge_orig_lerobot @ 0e9d76d07e9df3ea3eba257b2520d4913833fad2
- IPEC-COMMUNITY/fractal20220817_data_lerobot @ 91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e
- nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim @ 09c6de8af50168090e7e9cc01e1ec3bce788de24
No API tokens are passed to public download requests.
Staging: `/home/jigu/tmp/parcc-datasets-20260908/{widowx,fractal,gr1}`.
Source apparent sizes including excluded caches were about 21.7 GB, 22.0 GB,
134.8 GB respectively; local free space was 682 GB, target filesystem 3.4 TB.
Inventory resolved the GR1 size discrepancy: source has 24,000 extra .jpgpack
cache files. Excluding them, original dataset plus metadata is 41,460,717,592
bytes (~38.6 GiB). All 48,000 parquet/video files could be seeded from local
files with matching source byte sizes; PFS checksum verification is still required.
No source or staging data is automatically deleted. No rsync --delete.
Metadata is copied to PARCC first, followed by full payload.
Interrupted rsync transfers use a partial directory and retry up to three times.

## Verification and status

After payload: full-content rsync checksum dry-runs source -> staging and
staging -> PARCC must BOTH show zero differences. Any differences trigger
checksum-based repair and re-verification before completion.

Local live state: `{dataset}.status.json`; `transfer.log` and per-phase logs.
On completion: `{dataset}.complete.json` records file count, bytes, metadata
episode counts, source/target and verification method. Completion manifests
are also copied into PARCC `data/_setup_20260908/`.
Dataset directories may exist before completion; only a complete manifest
means that dataset has finished both transfer and verification.

Initial nohup launch did not persist; it was replaced with a user systemd unit:
parcc-datasets-20260908.service, independent of the interactive tool session.
The coordinator was restarted after adding the public-asset accelerator;
partial files and completed metadata were preserved. Use systemctl --user
status parcc-datasets-20260908 to inspect. In non-login shells, set
XDG_RUNTIME_DIR=/run/user/1001 and
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus for systemctl.
To resume after failure, run the same coordinator (it rejects concurrent copies
with flock). Keep this local source checkout and staging directory in place.

For GR1 training, use the 24 `data/gr1/gr1_unified.*/` task roots as the dataset
list, preserving the existing colon-separated multi-dataset convention.

## Last verified startup status (2026-09-08 23:57 UTC)

Both systemd services are active. All 26 metadata info.json files are present
on PARCC (24 GR1 tasks + WidowX + Fractal). GR1 original payload is staged
locally and doing PFS checksum synchronization; both other datasets have
started their inventory-restricted public-asset seed downloads. None has a
complete manifest yet. This is NOT a completed full-data transfer claim.
