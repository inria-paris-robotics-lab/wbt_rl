# Adding a module to WBT-RL

WBT-RL plugs external solutions into a four-stage pipeline — **retargeting → training → inference →
deployment** — without modifying their code. A "module" is one solution wired into one stage (a retargeter,
a trainer, an inference engine, or a deployer).

This guide is the **single end-to-end checklist** for adding one. Each numbered step is an independent,
coordinated place the new module touches. None of them requires editing `scripts/` — the orchestrators read
everything from `cfg/` at runtime.

> Throughout, `NN_stage` is one of `01_retargeting`, `02_training`, `03_inference`, `04_deployment`, and
> `<module>` is your new module's short name (e.g. `gmr`, `holosoma_custom`).

---

## 1 — Register the upstream code as a submodule

External code stays untouched, pulled in as a git submodule.

```bash
git submodule add <upstream-url> modules/third_party/<module>
```

Then expose it to the relevant stage(s) with a **relative symlink** — the pipeline looks under
`modules/NN_stage/`, never under `third_party/` directly:

```bash
ln -s ../third_party/<module>/<path/to/entrypoint/pkg> modules/01_retargeting/<module>
```

A solution that spans several stages (like `holosoma`, which retargets *and* trains *and* infers) gets one
symlink per stage, all pointing into the same submodule. A self-contained solution (like `GMR`) can be the
submodule directly under `modules/01_retargeting/`.

Commit `.gitmodules` and the new gitlink/symlinks together.

## 2 — Declare the module in `cfg/`

Create `cfg/NN_stage/<module>.yaml`. This file is the **only** point of contact between WBT-RL and the
module: it declares the conda env, the command, and how WBT-RL's arguments map to the module's CLI. Mirror
an existing yaml in the same stage directory — the schema is deliberately per-stage, not global.

Typical fields: `env`, `cmd`, `args`, `native_input_format` / `native_output_format`, plus stage-specific
blocks (`robot_config`, `simulators:`, `robot_exp_map`, …). A yaml may use `base: <other_module>` to inherit
and override another yaml in the same stage (this is how `holosoma_custom` extends `holosoma`).

See [cfg/README.md](cfg/README.md) for the field reference.

## 3 — Teach the adapter the module's data formats

In `src/motion_convertor/`:

1. **Register format names** in [`formats.py`](src/motion_convertor/formats.py) — add any new
   `native_input_format` / `native_output_format` strings to `KNOWN_FORMATS`. `validate_format()` rejects
   anything unregistered, so this is what makes a new format "exist".
2. **Add the converter(s)** — write the conversion code in the right role folder
   (`_to_retargeter_input/`, `_to_unified_input/`, `_to_unified_output/`, or `_to_trainer_input/`), then
   wire it in [`connectors.py`](src/motion_convertor/connectors.py): add a thin dispatch function (lazy-import
   the converter) and register its `(src_fmt, dst_fmt)` pair in the `CONNECTORS` table. A no-op conversion
   uses `_identity`.
3. **If conversion must run in a foreign conda env** (e.g. the module's own retargeting env), put the heavy
   work in a script under `wrappers/` and invoke it through `conda_run()` / `run_entry_point()` from
   [`_subprocess.py`](src/motion_convertor/_subprocess.py), rather than importing it in-process.

Mind the two conversion philosophies (see
[motion_convertor README](src/motion_convertor/README.md#two-connector-philosophies-by-design)): the
**retargeting path goes through the unified format** (so a new retargeter needs only `→ unified` and
`→ retargeter-input` converters), while the **training path uses direct `(retargeter, trainer)` pairs**.
A structurally new trainer therefore needs one trainer-input converter per retargeter that feeds it.

## 4 — Add an installer

Create `installers/pipe/<module>.sh` (WBT-managed env) or `installers/modules/<module>.sh` (module-managed
env), reusing the helpers in [`installers/lib.sh`](installers/lib.sh) (`_ensure_conda`, `_create_env`,
`_uv_pip`, …). To make it part of `./install.sh` (the `all` target), add one line to
[`installers/MODULES`](installers/MODULES):

```
modules/<module>
```

Leaving the module **out** of `MODULES` keeps it installable on demand (`./install.sh <module>`) but excluded
from a full install — this is exactly how the experimental `test_pipe` module is wired.

## 5 — Document it

- Add a row to the relevant table in [`README.md`](README.md) (Retargeters / Trainers / Inference /
  Deployment), with a status marker if it is experimental or WIP.
- If it introduces new datasets, robots, or flags, update [`scripts/README.md`](scripts/README.md).

---

## Checklist

- [ ] Submodule added under `modules/third_party/` (or `modules/NN_stage/`) + stage symlink(s), `.gitmodules` updated
- [ ] `cfg/NN_stage/<module>.yaml` created
- [ ] New formats added to `formats.py` (`KNOWN_FORMATS`)
- [ ] Converter(s) added under `_to_*/` and registered in `connectors.py` (`CONNECTORS`)
- [ ] Foreign-env conversions go through a `wrappers/` script + `_subprocess.py`
- [ ] `installers/<...>/<module>.sh` written; line added to `installers/MODULES` (unless intentionally manual)
- [ ] `README.md` (and `scripts/README.md` if needed) updated

No change to `scripts/retarget.py`, `train.py`, `infer.py`, or `deploy.py` should be necessary — if it is,
the abstraction is leaking and the config/adapter layers should absorb it instead.
