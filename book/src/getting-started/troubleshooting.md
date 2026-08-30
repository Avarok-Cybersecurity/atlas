# Troubleshooting

Problems that stop `atlasctl run` before a model ever loads, and what to do
about each. If your problem is a model that starts and then misbehaves, the
[Quickstart](./quickstart.md) has a section for that.

## `permission denied` talking to Docker

```text
permission denied while trying to connect to the Docker daemon socket
at unix:///var/run/docker.sock
```

**What it means.** Docker is installed and running. The daemon answered your
request and refused it, because your user is not in the `docker` group. This is
the single most common reason a fresh DGX Spark cannot launch a model, and it
has nothing to do with the hardware.

**The fix, once:**

```sh
sudo usermod -aG docker $USER
newgrp docker          # or log out and back in
```

`usermod` changes your groups; it does not change the groups of a shell that is
already running. `newgrp docker` starts a shell that has the new membership, and
logging out and back in achieves the same thing for every shell.

Confirm it took:

```sh
docker info --format '{{.ServerVersion}}'
```

If that prints a version, `atlasctl run` will work. You do not need to restart
the Atlas agent — it re-checks its own capability, so the control plane stops
saying "this machine cannot run models" within a few seconds.

### Do not use `sudo atlasctl`

It appears to work, and it is the wrong move:

- the model runs as root, and so does everything the container does;
- `~/.atlas` collects root-owned files that your normal user then cannot read,
  so the next unprivileged `atlasctl run` fails in a way that looks unrelated;
- `sudo` uses root's `PATH`, so `atlasctl` is frequently "not found" even though
  `which atlasctl` finds it for you.

Fix the group membership instead. It is a one-time change.

### Rootless Docker and Podman

If you run rootless Docker, the socket is under `$XDG_RUNTIME_DIR` rather than
`/var/run`, and group membership is not the issue — check that
`DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock` is exported. Docker's own
[post-installation guide][post-install] is the canonical reference for both the
group and the rootless setups.

[post-install]: https://docs.docker.com/engine/install/linux-postinstall/

## `Cannot connect to the Docker daemon`

```text
Cannot connect to the Docker daemon at unix:///var/run/docker.sock.
Is the docker daemon running?
```

Nothing is listening. Start it:

```sh
sudo systemctl start docker
sudo systemctl enable docker    # so it survives a reboot
```

## `docker: command not found`

Docker is not installed, or not on this shell's `PATH`. `atlasctl list`,
`atlasctl show` and `atlasctl run --print` all work without a container engine —
only `atlasctl run` needs one. See [Installation](./installation.md) for the
prerequisites, including the NVIDIA container runtime.

## `atlasctl` is not on PATH

The installer puts the binary in `~/.local/bin` and tells you if that directory
is not on your `PATH`. Add it:

```sh
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
```

Note the absence of a trailing slash. A `PATH` entry that ends in `/` is legal
and works, but it makes `which` print a doubled slash —
`/home/you/.local/bin//atlasctl` — which looks like a bug and is not one:
`which` joins the `PATH` entry to the program name without checking whether the
entry already ends in a separator.

## The control plane says this machine cannot run models

The browser is repeating what the local agent told it. The agent decides by
running `docker info`, so the cause is almost always one of the Docker problems
above — most often the permission one. Fix that and the page corrects itself
within a few seconds; there is no need to restart the agent.

To see the agent's own view:

```sh
atlasctl doctor
```

`doctor` reports each check and exits non-zero if any of them found a problem,
so it is safe to use in a script.
