# Start flash-4 at boot (systemd), and stop it again

Running `./flash-4/start.sh` by hand is fine until the machine reboots (kernel updates, power cycle, the DGX Spark
being moved). This page covers starting the server automatically at boot, and every way to stop it afterwards.

Written for this machine: Ubuntu 24.04.5 (systemd 255, PID 1), one GB10 GPU, Docker 29.6.2 with `docker.service`
enabled, and the `gb10` account in the `docker` group but **not** passwordless sudo. Paths assume
`~/projects/tensorfold/flash-4`. The unit is at `flash-4/systemd/tensorfold-flash-4.service`; that file is the source
of truth, this page explains it.

## The short version

```bash
# install (asks for your password once; starts nothing now)
~/projects/tensorfold/flash-4/systemd/install-autostart.sh --dry-run   # prints the three commands it would run
~/projects/tensorfold/flash-4/systemd/install-autostart.sh             # installs and enables it for the next boot

# want it running before the next reboot?
sudo systemctl start tensorfold-flash-4
journalctl -u tensorfold-flash-4 -f           # the model itself needs ~3 min
curl -s localhost:8888/health

# stop it now (it still starts at the next boot)
sudo systemctl stop tensorfold-flash-4
# stop it now *and* on every later boot
sudo systemctl disable --now tensorfold-flash-4
# or just: stop the container, exactly as before, without root
~/projects/tensorfold/flash-4/end.sh

# and to take the boot service out again (leaves a running server alone)
~/projects/tensorfold/flash-4/systemd/uninstall-autostart.sh
```

`systemd/install-autostart.sh` is three commands with safety checks: it refuses to overwrite an installed unit that
differs from the repo copy, and refuses to install a unit whose hardcoded `WorkingDirectory`/`User=` belong to a
different checkout. Add `--now` to install *and* start; that restarts a server you started by hand, because the unit's
pre-start step runs `end.sh`, which cuts whatever is in flight after its grace period. Without `--now` the running
container is untouched — it goes away at the reboot on its own (`docker run --rm`) and the service owns the next one.

## Why `docker update --restart=unless-stopped` is not enough

That is the usual answer for a container, and it does nothing here. `` always runs
`docker run --rm`, so the container is **deleted** when the daemon stops — there is no container left for dockerd to
relaunch. The patch set, the mounted `tools/*.py`, the `TENSORFOLD_*` environment and the HF-cache snapshot lookup all
live in `start.sh`, so something has to re-run `start.sh` at boot. Three ways, in decreasing order of sanity:

| | Root needed | Starts without login | Supervised / restarts on crash | Ordering after dockerd |
|---|---|---|---|---|
| **A. system unit** (recommended) | yes, once | yes | yes | yes (`After=docker.service`) |
| B. user unit + linger | no (polkit prompt for linger) | yes | yes | no, only the wait loop |
| C. cron `@reboot` | no | yes | no | no, only the wait loop |

## Before enabling it

- **`.env` is read by `start.sh`, not by `end.sh`.** `end.sh` only knows its own defaults (`NAME=tf-flash-4`,
  `PORT=8888`). If you set `NAME` or `PORT` in `.env`, add the same `Environment=NAME=…` / `Environment=PORT=…` lines
  to the unit, otherwise the pre-start cleanup misses the container and `start.sh` then exits with
  `container … already exists`.
- **The GPU is effectively exclusive.** The defaults (`PARALLEL=9`, `CONTEXT=262144`, int8 KV) reserve ~75 GiB of 115.
  Do not autostart a second model recipe next to this one; whoever boots first wins the memory.
- **`HOST=0.0.0.0` (the default) exposes an unauthenticated API on the LAN from the moment the machine boots.** Set
  `HOST=127.0.0.1` in `.env` if this box is not on a trusted network.
- **Prerequisites must be on the local disk and complete**, since nobody is there to answer a prompt: the pack and the
  drafter in `~/.cache/huggingface` (no `*.incomplete` blobs), and `vision-f16-4.05.safetensors` for vision. `start.sh`
  checks all of it and exits with a message you will find in the journal.
- **Decide what `FOREGROUND` means to you.** The unit sets `Environment=FOREGROUND=1`, which overrides `FOREGROUND=0`
  in `.env` (real environment variables win over `.env` in `start.sh`). That is what makes `systemctl stop` stop the
  container; see below.

## A. The system service

By hand, which is exactly what the installer does:

```bash
sudo cp ~/projects/tensorfold/flash-4/systemd/tensorfold-flash-4.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tensorfold-flash-4.service      # for the next boot; use enable --now to start it as well
```

Then reboot once and confirm, since a boot-time environment is not the same shell as this one:

```bash
sudo reboot
journalctl -b -u tensorfold-flash-4 | tail -40      # after it comes back
systemctl status tensorfold-flash-4 --no-pager
curl -s localhost:8888/health
```

Every directive in the unit exists because of something in these scripts:

| Directive | Why |
|---|---|
| `User=gb10`, `HOME=/home/gb10` | `start.sh` resolves the pack through `$HOME/.cache/huggingface` and needs the `docker` group, so it must not run as root. |
| `Environment=FOREGROUND=1` | `start.sh` then runs the container without `-d`, so the foreground `docker run` stays as systemd's main process and `systemctl stop` sends it SIGTERM, which stops the container (its `--rm` removes it, GPU memory freed). |
| `ExecStartPre=…timeout 600 …until docker info && nvidia-smi -L…` | dockerd and the GPU stack can come up after this unit; wait up to 10 min instead of failing once. No `$` shell variables in it on purpose — systemd expands `$FOO` in `Exec*` lines itself. |
| `ExecStartPre=-…/end.sh` | `start.sh` refuses to run while the container name exists. The leading `-` ignores the case where nothing is there. |
| `TimeoutStartSec=infinity` | a changed patch set rebuilds a 24.6 GB image before the server even starts. |
| `TimeoutStopSec=90` | systemd's own patience before it SIGKILLs the cgroup. |
| `Restart=on-failure`, `RestartSec=60` | the server dies → retry in a minute. Deliberate `systemctl stop` never triggers it. |
| `SuccessExitStatus=143` | stopping the container by hand makes `docker run` exit 143 (SIGTERM). Without this, `Restart=on-failure` reads your own `end.sh` as a crash and relaunches the server 60 s later. |
| `StartLimitIntervalSec=1800`, `StartLimitBurst=5` | a permanently broken setup (pack deleted, bad `.env`) gives up after 5 tries per half hour instead of hot-looping forever. |
| `SyslogIdentifier=tensorfold-flash-4` | greppable logs: `journalctl -u tensorfold-flash-4`. |

Two optional tweaks. The first is commented out in the unit already:

```ini
# only report "started" once /health answers, so a failed load counts as a failure and gets retried
ExecStartPost=/usr/bin/timeout 3600 /bin/sh -c 'until curl -sf --max-time 5 localhost:8888/health >/dev/null; do sleep 5; done'
```

…with `TimeoutStartSec=3600` instead of `infinity` (with `Type=simple`, systemd otherwise calls the service "running"
the moment `start.sh` is forked, long before the model is loaded).

The container is started with `--stop-timeout 30`, so `systemctl stop` is as gentle as `end.sh`; keep `TimeoutStopSec` above it.

## Stopping it

```bash
sudo systemctl stop tensorfold-flash-4     # stop now; still starts at the next boot
~/projects/tensorfold/flash-4/end.sh       # same result, no root, 30 s grace + a warning about live requests
sudo systemctl disable --now tensorfold-flash-4   # stop now and on every later boot
sudo systemctl mask tensorfold-flash-4     # nothing may start it, not even a dependency
sudo systemctl unmask tensorfold-flash-4
```

`stop` alone does **not** survive a reboot — an enabled unit starts again. Use `disable` for that. With
`SuccessExitStatus=143` in place, `end.sh` (or `docker stop tf-flash-4`) also leaves the unit cleanly `inactive`
instead of restarting it.

Check what state you are in:

```bash
systemctl is-active tensorfold-flash-4      # active | inactive | failed
systemctl is-enabled tensorfold-flash-4     # enabled | disabled | masked
docker ps -a --filter name=tf-flash-4
nvidia-smi                                  # memory actually back
```

Root is unavoidable for `systemctl` on a system unit: Ubuntu's polkit policy for
`org.freedesktop.systemd1.manage-units` is `auth_admin*`, and admin means the `sudo`/`admin` group, so you
authenticate with your own password (cached briefly). `end.sh` skips polkit entirely through your `docker` group
membership.

If a stop leaves debris (unit `failed`, container still listed):

```bash
~/projects/tensorfold/flash-4/end.sh
sudo systemctl reset-failed tensorfold-flash-4
```

To remove the installation completely, use the reverse of the installer:

```bash
~/projects/tensorfold/flash-4/systemd/uninstall-autostart.sh --dry-run
~/projects/tensorfold/flash-4/systemd/uninstall-autostart.sh
```

It disables the unit, deletes the unit file and any `systemctl edit` drop-in directory, reloads systemd and clears a
`failed` result. **It leaves a running server alone** — the container keeps serving until you run `end.sh` — and it
touches no model, image or kernel cache. Options: `--stop` to disable *and* stop in one step (10 s grace, live
requests get cut), `--force` to delete an installed unit that has drifted from the repo copy (without it you get a diff
and an exit instead of a deletion), `--dry-run` to see the commands. Unmasking happens automatically if the unit is
masked. By hand it is:

```bash
sudo systemctl disable --now tensorfold-flash-4     # without --now if the server should keep running
sudo rm -rf /etc/systemd/system/tensorfold-flash-4.service.d
sudo rm -f /etc/systemd/system/tensorfold-flash-4.service
sudo systemctl daemon-reload && sudo systemctl reset-failed tensorfold-flash-4
```

Nothing else in the recipe references the unit, so the repo copies under `systemd/` are all that is left, and
`install-autostart.sh` puts it back.

## B. User service, no root

Same body, without `User=`/`Group=`, in `~/.config/systemd/user/tensorfold-flash-4.service`, and `WantedBy=default.target`.

```bash
systemctl --user daemon-reload
systemctl --user enable --now tensorfold-flash-4.service
loginctl enable-linger gb10                       # start the user manager at boot, not at login
loginctl show-user gb10 --property=Linger         # Linger=yes
```

Keep the `until docker info … nvidia-smi` wait loop: a user manager cannot order itself `After=docker.service`,
because that unit belongs to a different manager. Lingering is what makes this run without anyone logging in;
without it the unit only starts when you log in. `enable-linger` itself usually asks for polkit authentication, and the
service is invisible to `systemctl status` / `journalctl -u` on the system manager — use `systemctl --user status …`
and `journalctl --user -u …` instead (this box keeps `/var/log/journal`, so those lines do survive a reboot).

## C. cron `@reboot`

```cron
@reboot /usr/bin/timeout 900 /bin/sh -c 'until docker info >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; do sleep 5; done'; /home/gb10/projects/tensorfold/flash-4/end.sh; /home/gb10/projects/tensorfold/flash-4/start.sh >> /home/gb10/projects/tensorfold/flash-4/runtime/boot.log 2>&1
```

The lowest-effort option and the weakest: nothing restarts the server if it dies, nothing stops it cleanly at shutdown
(it goes down when dockerd does), and the log is a file you rotate yourself. Fine as a stopgap.

## Logs

With `FOREGROUND=1` the model's stdout goes to **both** journald (`journalctl -u tensorfold-flash-4`) and the
container's own log driver, so `docker logs -f tf-flash-4` keeps working while the service owns it. If the
duplication matters, cap the container side with `--log-opt max-size=50m` in `start.sh`'s `docker run`, and note that
journald keeps its own quota (`journalctl --disk-usage`).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Unit `failed` right after boot, journal shows `revision … is not in …` or `*.incomplete` | the model folder is missing or the cache moved | rebuild the model folder with `./flash-4/build_model_dir.sh` (it uses only files already in the cache), then `sudo systemctl start tensorfold-flash-4` |
| `container tf-flash-4 already exists` | a `NAME`/`PORT` in `.env` that `end.sh` cannot see, or a container from a detached start | mirror `NAME` in the unit, or `end.sh` once by hand |
| Starts, then is back a minute after you stopped it | `end.sh`'s 143 exit counted as a crash | keep `SuccessExitStatus=143` in the unit |
| Unit sits in `activating` for ~10 min, then `Control process exited, code=exited, status=124` | the pre-start loop never saw dockerd or `nvidia-smi -L` | `systemctl status docker nvidia-persistenced && docker info`; the wait only gives them time, it cannot fix a broken daemon |
| Service is `active` but port 8888 refuses | still loading (~3 min, longer after a patch change) | `docker logs -f tf-flash-4` |
| A rebuild of the image on every boot | `scripts/prepare.sh` rebuilds only when `patches/` or its Dockerfile change (the image carries their hash as a label) | `docker image inspect -f '{{index .Config.Labels "tf.patches"}}' tensorfold-flash4:zig-02b324a`; a boot without network cannot rebuild, so build once beforehand |
| Server up at boot but too slow for the first minutes | the first start after a kernel/driver update compiles kernels | leave it; `~/.local/state/tf-flash-4/tensorfold` keeps what it learns |
