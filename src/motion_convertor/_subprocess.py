"""
Helpers for running module commands in their own conda environments.
Reads cfg/NN_stage/ and cfg/motion_convertor/ yamls to get env names and command templates.
"""
import copy
import functools
import os
import subprocess
import yaml
from pathlib import Path

from ._config import repo_root


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    - Nested dicts are merged recursively.
    - Lists and scalars in override replace those in base.
    - base is never mutated.
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@functools.lru_cache(maxsize=None)
def load_module_cfg(stage: str, module: str) -> dict:
    """Load cfg/<stage>/<module>.yaml, merging from `base:` if present.

    stage is a cfg/ subdirectory name (e.g. "01_retargeting", "motion_convertor").
    The returned dict is cached and shared across callers — do not mutate it.
    """
    cfg_path = repo_root() / "cfg" / stage / f"{module}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"cfg/{stage}/{module}.yaml: expected mapping, got {type(cfg).__name__}")
    if "base" in cfg:
        base_cfg = load_module_cfg(stage, cfg.pop("base"))
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def conda_run(
    env: str,
    cmd: str,
    cwd: Path | None = None,
    check: bool = True,
    interactive: bool = False,
    prefix: str | None = None,
    env_vars: dict | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a shell command inside a conda environment.

    Uses `conda run -n {env} --no-capture-output {cmd}` so that the
    subprocess inherits stdout/stderr (visible to the caller).

    When interactive=True, stdin is inherited from the calling terminal so
    that blocking prompts (e.g. input()) work correctly.
    """
    if cwd is None:
        cwd = repo_root()

    # `conda run` does not respect the calling shell's cwd — prepend an explicit cd.
    # Use --prefix when the env lives outside the active conda root (e.g. holosoma envs).
    if prefix is not None:
        env_selector = f"--prefix {os.path.expandvars(prefix)}"
    else:
        env_selector = f"-n {env}"

    env_prefix = ""
    if env_vars:
        env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items()) + " "

    full_cmd = f"conda run {env_selector} --no-capture-output bash -c 'cd {cwd} && {env_prefix}{cmd}'"
    stdin = None if interactive else subprocess.DEVNULL
    return subprocess.run(
        full_cmd,
        shell=True,
        check=check,
        stdin=stdin,
    )


def run_entry_point(stage: str, module: str, entry: str, args: dict, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """
    Run a named entry point from cfg/<stage>/<module>.yaml.

    `args` is a dict mapping WBT-RL arg names → values.
    Unrecognised keys are silently ignored (some entry points have no args).
    """
    cfg = load_module_cfg(stage, module)
    env = cfg["env"]
    ep = cfg["entry_points"][entry]

    cmd = os.path.expandvars(ep["cmd"])
    arg_map = ep.get("args", {})
    for wbt_key, value in args.items():
        flag = arg_map.get(wbt_key)
        if flag is not None:
            cmd += f" {flag} {value}"

    # Use per-entry cwd if specified in yaml, otherwise fall back to caller-supplied cwd
    ep_cwd = ep.get("cwd")
    if ep_cwd is not None:
        effective_cwd = repo_root() / ep_cwd
    elif cwd is not None:
        effective_cwd = cwd
    else:
        effective_cwd = repo_root()

    prefix = cfg.get("env_prefix")
    return conda_run(env, cmd, cwd=effective_cwd, prefix=prefix)
