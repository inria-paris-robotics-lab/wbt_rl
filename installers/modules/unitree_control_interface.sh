#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)/lib.sh"

install_unitree_control_interface() {
  _header "deployment — unitree_ros2 + unitree_control_interface"

  local WS="$REPO_ROOT/modules/04_deployment/unitree_ros2/cyclonedds_ws"
  local SRC="$WS/src"
  local UCI_DIR="$SRC/unitree_control_interface"
  local UCI_ENV="unitree_control_interface"
  local ENV_ROOT="$WBT_CONDA_ROOT/envs/$UCI_ENV"
  local ENV_PYTHON="$ENV_ROOT/bin/python"
  local SENTINEL="$HOME/.wbt_deps/.env_setup_finished_$UCI_ENV"

  _ensure_conda "$WBT_CONDA_ROOT" "$HOME/.wbt_deps"

  git -C "$REPO_ROOT" submodule update --init modules/04_deployment/unitree_ros2

  if [[ ! -d "$UCI_DIR" ]]; then
    echo "  Cloning unitree_control_interface..."
    git clone -b watchdog-logging https://github.com/inria-paris-robotics-lab/unitree_control_interface.git \
      --recursive "$UCI_DIR"
  else
    _ok "unitree_control_interface already cloned"
  fi

  if [[ -f "$SENTINEL" ]]; then
    _ok "unitree_control_interface env already installed (sentinel found)"
    return
  fi

  if [[ ! -d "$ENV_ROOT" ]]; then
    echo "  Creating conda env '$UCI_ENV' in ~/.wbt_deps/..."
    MAMBA_ROOT_PREFIX="$WBT_CONDA_ROOT" "$WBT_CONDA_ROOT/bin/mamba" env create \
      --prefix "$ENV_ROOT" \
      -f "$UCI_DIR/environment.yaml" -v
  else
    _ok "conda env '$UCI_ENV' already exists"
  fi

  local MAMBA_BIN="$WBT_CONDA_ROOT/bin/mamba"

  _CURRENT_PY="$("$ENV_ROOT/bin/python" --version 2>/dev/null | awk '{print $2}' | cut -d. -f1,2)"
  if [[ "$_CURRENT_PY" != "3.11" ]]; then
    echo "  Pinning python=3.11 (currently $_CURRENT_PY)..."
    "$MAMBA_BIN" install -y python=3.11 -c conda-forge -c robostack-staging --override-channels --prefix "$ENV_ROOT"
  else
    _ok "python already 3.11, skipping pin"
  fi

  _CURRENT_CMAKE="$("$ENV_ROOT/bin/cmake" --version 2>/dev/null | awk 'NR==1{print $3}' | cut -d. -f1,2)"
  if [[ "$_CURRENT_CMAKE" != "3.28" ]]; then
    echo "  Pinning cmake=3.28 (rosidl_generator_py compatibility)..."
    "$MAMBA_BIN" install -y cmake=3.28 -c conda-forge -c robostack-staging --override-channels --prefix "$ENV_ROOT"
  else
    _ok "cmake already 3.28, skipping pin"
  fi

  if [[ ! -f "$ENV_ROOT/lib/liblttng-ust.so" ]] && ! "$ENV_ROOT/bin/conda" list -p "$ENV_ROOT" lttng-ust 2>/dev/null | grep -q lttng-ust; then
    echo "  Installing lttng-ust (rclcpp tracing backend)..."
    "$MAMBA_BIN" install -y lttng-ust -c conda-forge -c robostack-staging --override-channels --prefix "$ENV_ROOT"
  else
    _ok "lttng-ust already installed, skipping"
  fi

  if ! "$ENV_ROOT/bin/vcs" --version &>/dev/null; then
    # Use conda-forge vcstool: conda's setuptools 82+ doesn't expose
    # pkg_resources, which breaks pip-installed vcstool's entry point.
    MAMBA_ROOT_PREFIX="$WBT_CONDA_ROOT" "$WBT_CONDA_ROOT/bin/mamba" install \
      -y --prefix "$ENV_ROOT" vcstool -c conda-forge
    "$ENV_PYTHON" -m pip install setuptools
  fi
  git config --global url."https://github.com/".insteadOf "git@github.com:"
  _vcs_status=0
  (cd "$SRC" && "$ENV_ROOT/bin/vcs" import --recursive --skip-existing < "$UCI_DIR/git-deps.yaml") || _vcs_status=$?
  git config --global --unset url."https://github.com/".insteadOf
  [[ $_vcs_status -eq 0 ]] || exit $_vcs_status

  (
    set +u
    export PATH="$ENV_ROOT/bin:$PATH"
    export Python_ROOT_DIR="$ENV_ROOT"
    export Python3_ROOT_DIR="$ENV_ROOT"
    _PY_VER="$("$ENV_ROOT/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    export PYTHONPATH="$ENV_ROOT/lib/python${_PY_VER}/site-packages${PYTHONPATH:+:$PYTHONPATH}"

    if [[ -f "$ENV_ROOT/setup.bash" ]]; then
      source "$ENV_ROOT/setup.bash"
    fi

    cd "$WS"
    CMAKE_ARGS=(
      "-DPython_ROOT_DIR=$ENV_ROOT"
      "-DPython3_ROOT_DIR=$ENV_ROOT"
      "-DPython_EXECUTABLE=$ENV_ROOT/bin/python"
      "-DPython3_EXECUTABLE=$ENV_ROOT/bin/python"
      "-DPYTHON_EXECUTABLE=$ENV_ROOT/bin/python"
      "-DCMAKE_CXX_FLAGS=-DTRACETOOLS_DISABLED"
    )

    colcon build --packages-select cyclonedds --cmake-args "${CMAKE_ARGS[@]}"
    source install/setup.bash

    (
      export PATH="$ENV_ROOT/bin:$PATH"
      export CYCLONEDDS_HOME="$WS/install/cyclonedds"
      "$ENV_PYTHON" -m pip install "cyclonedds==0.10.5" --no-binary :all: --force-reinstall
    )

    colcon build --packages-skip unitree_sdk2py --cmake-args "${CMAKE_ARGS[@]}"
  )

  (
    set +u
    export PATH="$ENV_ROOT/bin:$PATH"
    cd "$WS"
    source install/setup.bash
    export CYCLONEDDS_HOME="$WS/install/cyclonedds"
    "$ENV_PYTHON" -m pip install -e "$SRC/unitree_sdk2_python"
  )

  touch "$SENTINEL"
  _ok "deployment stack installed — env: $UCI_ENV"
  echo ""
  echo "  To use:"
  echo "    conda activate $UCI_ENV"
  echo "    source $WS/install/setup.bash"
}

install_unitree_control_interface "$@"
