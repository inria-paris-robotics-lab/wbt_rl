# Shared helpers and vars for all WBT-RL installers.
# Sourced by each installer script — not executed directly.

REPO_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &>/dev/null && pwd )"
GMR_DIR="$REPO_ROOT/modules/01_retargeting/GMR"
HOLOSOMA_SCRIPTS="$REPO_ROOT/modules/third_party/holosoma_custom/scripts"
TEST_PIPE_SCRIPTS="$REPO_ROOT/modules/third_party/test_pipe/scripts"

WBT_CONDA_ROOT="$HOME/.wbt_deps/miniconda3"
HOLOSOMA_CONDA_ROOT="$HOME/.holosoma_deps/miniconda3"
TEST_PIPE_CONDA_ROOT="$HOME/.test_pipe_deps/miniconda3"

_header() { echo ""; echo "══════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════"; }
_ok()     { echo "  ✓ $1"; }

_ensure_conda() {
  local root="$1" deps_dir="$2"
  if [[ ! -d "$root" ]]; then
    _header "Bootstrapping miniforge → $root"
    mkdir -p "$deps_dir"
    local os arch installer
    os="$(uname -s)"; arch="$(uname -m)"
    if   [[ "$os" == "Linux"  && "$arch" == "aarch64" ]]; then installer="Miniforge3-Linux-aarch64.sh"
    elif [[ "$os" == "Linux"  ]];                          then installer="Miniforge3-Linux-x86_64.sh"
    elif [[ "$os" == "Darwin" && "$arch" == "arm64"   ]]; then installer="Miniforge3-MacOSX-arm64.sh"
    elif [[ "$os" == "Darwin" ]];                          then installer="Miniforge3-MacOSX-x86_64.sh"
    else echo "ERROR: unsupported OS: $os" >&2; exit 1; fi
    local tmp="$deps_dir/miniforge_install.sh"
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/$installer" -o "$tmp"
    bash "$tmp" -b -u -p "$root" && rm "$tmp"
  fi
  "$root/bin/conda" config --system --add channels conda-forge
  "$root/bin/conda" config --system --set channel_priority strict
  "$root/bin/conda" config --system --remove channels defaults 2>/dev/null || true
  [[ -f "$root/bin/mamba" ]] || \
    "$root/bin/conda" install -y mamba -c conda-forge -n base --override-channels
}

_create_env() {
  local root="$1" name="$2" python="${3:-3.11}"
  local env_root="$root/envs/$name"
  [[ -d "$env_root" ]] && { _ok "env '$name' already exists"; return; }
  "$root/bin/mamba" create -y --prefix "$env_root" python="$python" \
    -c conda-forge --override-channels
}

_uv_pip() {
  local env_root="$1"; shift
  [[ -f "$env_root/bin/uv" ]] || "$env_root/bin/python" -m pip install uv
  UV_HTTP_TIMEOUT=300 "$env_root/bin/uv" pip install --python "$env_root/bin/python" --system "$@"
}

_clean_bash_pinned() {
  local envs_dir="$1"; shift
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
      -u CONDA_EXE -u _CONDA_EXE -u CONDA_PYTHON_EXE \
      -u CONDA_PROMPT_MODIFIER -u _CONDA_ROOT -u _CE_M -u _CE_CONDA \
      -u CONDARC -u CONDA_ENVS_PATH \
      CONDA_ENVS_PATH="$envs_dir" "$@"
}

_holosoma_prep_env() {
  local env_name="$1" python_ver="$2"; shift 2
  local env_root="$HOLOSOMA_CONDA_ROOT/envs/$env_name"

  if [[ ! -d "$env_root" ]]; then
    _ensure_conda "$HOLOSOMA_CONDA_ROOT" "$HOME/.holosoma_deps"
    "$HOLOSOMA_CONDA_ROOT/bin/mamba" create -y --prefix "$env_root" \
      python="$python_ver" "$@" -c conda-forge --override-channels
  fi

  [[ -f "$env_root/bin/uv" ]] || "$env_root/bin/python" -m pip install uv

  local uv_bin="$env_root/bin/uv"
  cat > "$env_root/bin/pip" <<PIPSHIM
#!/usr/bin/env bash
# Delegates to uv pip.
# Check if the command is install, to filter "pip install pip"
if [[ "\$1" == "install" ]]; then
  only_pip=1
  for a in "\${@:2}"; do
    if [[ -n "\$a" ]]; then
        [[ "\$a" =~ ^- ]] && continue
        [[ "\$a" =~ ^pip([>=<!@].*)?$ ]] && continue
        only_pip=0; break
    fi
  done
  [[ \$only_pip -eq 1 ]] && exit 0

  args=()
  for arg in "\$@"; do
    [[ -n "\$arg" ]] && args+=("\$arg")
  done
  exec "$uv_bin" pip install "\${args[@]:1}"
else
  exec "$uv_bin" pip "\$@"
fi
PIPSHIM
  chmod +x "$env_root/bin/pip"
  ln -sf pip "$env_root/bin/pip3"
}

_test_pipe_prep_env() {
  local env_name="$1" python_ver="$2"; shift 2
  local env_root="$TEST_PIPE_CONDA_ROOT/envs/$env_name"

  if [[ ! -d "$env_root" ]]; then
    _ensure_conda "$TEST_PIPE_CONDA_ROOT" "$HOME/.test_pipe_deps"
    "$TEST_PIPE_CONDA_ROOT/bin/mamba" create -y --prefix "$env_root" \
      python="$python_ver" "$@" -c conda-forge --override-channels
  fi

  [[ -f "$env_root/bin/uv" ]] || "$env_root/bin/python" -m pip install uv

  local uv_bin="$env_root/bin/uv"
  cat > "$env_root/bin/pip" <<PIPSHIM
#!/usr/bin/env bash
if [[ "\$1" == "install" ]]; then
  only_pip=1
  for a in "\${@:2}"; do
    if [[ -n "\$a" ]]; then
        [[ "\$a" =~ ^- ]] && continue
        [[ "\$a" =~ ^pip([>=<!@].*)?$ ]] && continue
        only_pip=0; break
    fi
  done
  [[ \$only_pip -eq 1 ]] && exit 0

  args=()
  for arg in "\$@"; do
    [[ -n "\$arg" ]] && args+=("\$arg")
  done
  exec "$uv_bin" pip install "\${args[@]:1}"
else
  exec "$uv_bin" pip "\$@"
fi
PIPSHIM
  chmod +x "$env_root/bin/pip"
  ln -sf pip "$env_root/bin/pip3"
}

_test_pipe_run() {
  local env_name="$1" script="$2"; shift 2
  _clean_bash_pinned "$TEST_PIPE_CONDA_ROOT/envs" \
    WORKSPACE_DIR="$HOME/.test_pipe_deps" \
    bash "$script" "$@"
}

_holosoma_run() {
  local with_sudo=0 extra_env=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --sudo) with_sudo=1; shift ;;
      *=*)    extra_env+=("$1"); shift ;;
      *)      break ;;
    esac
  done
  local env_name="$1" script="$2"; shift 2

  local fake_dir; fake_dir="$(mktemp -d)"

  if [[ $with_sudo -eq 1 ]]; then
    cat > "$fake_dir/sudo" <<'FAKESUDO'
#!/usr/bin/env bash
if [[ "$*" == *"apt"* ]]; then echo "[install.sh] skipping sudo apt"; exit 0; fi
exec /usr/bin/sudo "$@"
FAKESUDO
    chmod +x "$fake_dir/sudo"
  fi

  _clean_bash_pinned "$HOLOSOMA_CONDA_ROOT/envs" \
    WORKSPACE_DIR="$HOME/.holosoma_deps" \
    ${extra_env[@]+"${extra_env[@]}"} \
    PATH="$fake_dir:$PATH" \
    bash "$script" "$@"
  rm -rf "$fake_dir"
}
