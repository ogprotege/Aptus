# MLX-LM target-host acceptance, 2026-07-27

> **Status:** Passed target-runtime evidence | **Authority:** Immutable acceptance record | **Applies to:** Aptus 0.2 MLX-LM LoRA and QLoRA runtime path | **Audience:** Release reviewers and operators | **Last reviewed:** 2026-07-27 | **Review by:** On any MLX runtime, compiler, admission, or evidence-contract change

> **Result:** `measured-run-pass` in two clean, independent workflows
> **Host:** Apple M5 Pro, 64 GiB unified memory, macOS 26.6 (25G72)
> **Clean acceptance commit:** `36ef1314950c8f86a2b298e8d515e247734043ce`
> **Scope:** Runtime correctness and release acceptance. This is not a model-quality benchmark.

## Acceptance result

Aptus completed its generated MLX-LM QLoRA workflow twice against a real,
revision-pinned public model and the repository's synthetic support dataset.
Each workflow independently ran plan generation, bundle compilation, static
validation, dependency validation, model and data validation, measured
preflight, uninterrupted pilot, fresh-process adapter reload, confirmed full
training, final export, and a second fresh-process reload.

Both workflows ended in `measured-run-pass`.

| Clean run | Preflight ID | Pilot ID | Full run ID | Preflight peak | Pilot peak | Full peak |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | `bounded-smoke_c64b8c1d3f184112b1e3061a4840a99c` | `pilot_f9dfb07d8636412fbcecf50ad3333369` | `run_5ef5bc73bdff4a0abf428ca472be2789` | 524,594,168 B | 523,730,508 B | 582,055,054 B |
| 2 | `bounded-smoke_4b6a4755b4644d989543acdfa288e38e` | `pilot_1c328dda0c9f42669b6d87d3d4ff1088` | `run_d957e756c1074bbea0f177f553751f7e` | 524,491,768 B | 524,353,588 B | 581,975,426 B |

The generated input artifacts were byte-identical across both workflows:

- plan SHA-256: `02d9c45c73ff1240cc8fd981381066e9e09ed1e7d1a7e1b72a16b4aad9f4fe1f`;
- bundle ZIP SHA-256: `18a3ec97163ea1968137539e6045df5f092e2059c65da30c196676a6b05eda72`;
- bundle fingerprint: `9cf730c5e2c4044bc297742b1d0f4832ff992cbf4f0d1b6f63af6e2456680bbd`.

## Clean-checkout gate

The acceptance checkout started at
`b371f49ee7021f86c3271b5a57efcd958971ce08`. The recorded runtime-interpreter
fix was applied and committed only in the isolated checkout, producing
`36ef1314950c8f86a2b298e8d515e247734043ce`. `git status --porcelain=v1` was
empty before run 1, after run 1, and after run 2.

Each workflow used its own generated bundle, state directory, pilot output, and
full-run output. The external runtime environment and isolated model cache were
shared as immutable inputs. `aptus.__file__` resolved to the clean checkout.
The clean checkout remained unchanged throughout both workflows.

The earlier evidence at this directory's root records the exploratory shared-
checkout run that exposed the interpreter defect. The `clean-run-1/` and
`clean-run-2/` directories are the authoritative repeatability evidence.

## Deterministic training evidence

Both full runs produced the same:

- train losses: `[3.8527462482452393, 3.1543750762939453, 4.094791889190674]`;
- validation losses: `[4.8986897468566895, 3.818800449371338, 3.560231924057007]`;
- adapter delta L1: `5297.497747182846`;
- changed adapter tensors: `336`;
- optimizer updates: `3`;
- target instances: `168` across `24` transformer layers;
- final `adapters.safetensors` SHA-256:
  `4717543bb38f084573a6f1ea2fa0638d71c1a1a38b1b2103545951e052d5f31b`;
- final export size: `17,633,607` bytes;
- fresh-process generation: `4` tokens after pilot and full training.

Run-specific IDs, output paths, and timestamps intentionally differ. Those
values make the adapter config and artifact manifest hashes differ. The trained
adapter weights remain byte-identical.

## Bound model, data, and runtime

- Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- Immutable revision: `53a32aee5e9447773fd2b85988395066aef3700a`
- Model license: Apache-2.0
- Dataset: `examples/support-sft.jsonl`
- Dataset SHA-256: `bf2dca3d6398d639f47a883203920e1f52b0981becac96734147054e53f8aa44`
- Plan: `plan_df3b75050e6ea520a9c0`
- Candidate: `cand_f222b24ca5b139b67436`
- Compiler: `mlx-lm.qlora.v1`
- Python: `3.12.13`
- MLX: `0.31.2`
- MLX-LM: `0.31.3`

The model snapshot was downloaded at the exact revision. Python Hub requests
stalled after the host OS update, while direct HTTPS downloads remained
healthy. The managed attempt was cancelled cleanly. The exact files were then
downloaded, hashed, placed in an isolated Hugging Face cache, and used with
`HF_HUB_OFFLINE=1`. See `model-files.sha256` and `logs/model-download.log`.

Package-index requests showed the same post-update behavior. The external
Python environment was therefore populated from the previously resolved exact
environment. It lived outside the checkout and was not a host-global Aptus
environment.

## Timing and admission evidence

| Action | Clean run 1 | Clean run 2 |
| --- | ---: | ---: |
| Dependency | 1.07 s | 1.12 s |
| Model and data | 1.84 s | 1.86 s |
| Measured preflight | 5.00 s | 3.43 s |
| Uninterrupted pilot and reload | 6.01 s | 6.00 s |
| Confirmed full train and reload | 4.73 s | 5.06 s |

Full-train admission saw 35,438,772,224 bytes available in run 1 and
35,154,788,352 bytes in run 2. Both exceeded the 13,047,862,262-byte required
threshold, which includes an 8,589,934,592-byte reserve. The Aptus upper
estimate was 4,457,927,670 bytes.

No Aptus generated training, validation, reload, or run process remained at
handoff. No host-global Aptus GPU lease file was active. Adapter binaries were
not committed. Their names, sizes, and hashes are retained in each run's
`final-export.json` and `final-artifact-manifest.json`.

## Runtime-interpreter correction

The first managed dependency attempt exposed a product defect. Aptus resolved a
virtual environment's `bin/python` symlink to its base interpreter. The probe
saw the environment, but the recorded command later lost its installed
packages.

`src/aptus/runtime_env.py` now preserves the selected executable path while
still checking that the path exists, is a file, and is executable. A regression
test proves that a symlinked virtual-environment interpreter remains the
launched command. The exact change is retained in
`runtime-interpreter-fix.patch` with SHA-256
`c1307ba8835751b413f89ee8f56b98c03429fd5eb5429c8b909d8f8535b9295c`.

## Reproduction sequence

The generated bundle was operated from an external Python 3.12 environment:

```bash
export APTUS_MLX_PYTHON=/absolute/path/to/runtime-env/bin/python
export HF_HOME=/absolute/path/to/isolated-hf-home
export HF_HUB_OFFLINE=1

python -m aptus run BUNDLE --action dependency --state-dir STATE
python -m aptus run BUNDLE --action model-data --state-dir STATE
python -m aptus run BUNDLE --action preflight --state-dir STATE
python -m aptus run BUNDLE --action pilot --state-dir STATE
python -m aptus run BUNDLE --action train --confirm-full-train --state-dir STATE
```

`clean-run-1/logs/` and `clean-run-2/logs/` preserve the command and child
process output. `python-packages.txt` records the resolved environment.
`acceptance-summary.json` provides a machine-readable rollup. `SHA256SUMS`
binds every retained evidence file.

## Primary references

- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [MLX-LM fine-tuning guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [Pinned Qwen2.5 MLX model](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/53a32aee5e9447773fd2b85988395066aef3700a)
