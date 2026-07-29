# Aptus documentation drift audit — 2026-07-28

> **Status:** Archived | **Authority:** Immutable point-in-time audit record | **Applies to:** Documentation state between `e98ff55` and `51eea03` | **Audience:** Contributors and documentation reviewers | **Last reviewed:** 2026-07-29 | **Review by:** Never — this is a historical record and must not be rewritten

> **Recorded:** 2026-07-28 | **Baseline:** knowledge-graph build at 2026-07-27 15:15 | **Audited through:** PR #13 (`e98ff55`) | **Remediated by:** PR #14 (`51eea03`), merged 2026-07-29

Scope: every documentation claim made stale by the commits merged between the baseline and `e98ff55`.

Method: three rounds of parallel dimension auditors, each finding independently attacked by two
adversarial verifiers — one re-reading every cited line to kill fabricated citations, one attempting to
prove the mismatch was not real drift. A finding survived only if no verifier refuted it AND its citations
were re-verified. 42 agents, 65 candidate findings, 31 refuted, 34 confirmed (32 distinct locations).

**32 distinct confirmed findings across 12 root causes.**

## How to read this record

**Line numbers here are historical.** Every `file:line` citation describes the tree at `e98ff55`, before
remediation. PR #14 changed those exact lines, so a citation below will not point at the same text in the
current tree. That is expected: this is a point-in-time record, not a live defect list. Do not treat a stale
line number as an open finding.

Relative link paths appearing *inside* quoted document text were re-based to this file's location so the
repository link check resolves them. The quoted prose is otherwise verbatim.

**All 12 root causes were remediated in PR #14.** The audit itself modified no code or documentation; the
"Fix order" and per-finding "Fix" entries below record what was *recommended*, and all of it was
subsequently applied across 24 documentation files plus one component fix
(`web/src/components/ExpertTopologyRail.tsx`).

**The refuted findings are the most reusable part of this record.** Thirty-one candidate findings were
raised and killed — a 48% false-positive rate on first pass. Section "Findings" lists only survivors, but
the refutation reasoning is preserved in the per-round detail so that a future audit does not re-litigate
settled questions. Two recur often enough to name here: the UI/UX contract deliberately delegates visual
tokens to `web/src/styles.css` and `web/src/components/`, so the brand mark's absence from it is not drift;
and `capability-matrix.md` already carries the normative Qwen3 MoE admission table, so other documents are
not required to duplicate it.

---

## Fix order

Weighted by user harm: a doc that causes a wrong hardware or method decision outranks one that
inconveniences a contributor.

1. **A — MLX-LM is exempt from the device four-bit capability gate**
   Files: `docs/guides/choose-a-method.md`, `docs/guides/compare-plans.md`, `docs/product/current-capabilities.md`, `docs/reference/capability-matrix.md`, `docs/reference/method-registry.md`
2. **B — MLX unified-memory admission is now a live gate at model-data time**
   Files: `docs/guides/troubleshooting.md`, `docs/methodology/memory-estimation.md`, `docs/methodology/preflight-calibration.md`, `docs/operations/operator-checklist.md`, `docs/reference/validation-states.md`
3. **L — MLX-LM refuses whole-text rows, and no document says so**
   Files: `docs/guides/prepare-a-dataset.md`
4. **J — --reserve-gib is silently floored to 8.0 GiB under mps**
   Files: `docs/guides/model-dataset-hardware.md`, `docs/reference/cli.md`
5. **C — model-data-evidence.json is now a tolerated mutable bundle path**
   Files: `docs/architecture/data-and-identity-flow.md`, `docs/operations/state-storage-retention.md`, `docs/reference/bundle-manifest.md`
6. **G — MLX quantized storage is priced from the bound quantization_layout**
   Files: `docs/methodology/memory-estimation.md`
7. **K — The expert routing rail drops evidence on the runtime-mismatch branch**
   Files: `docs/product/ui-ux.md`
8. **E — MLX preflight.py is a 23-line platform check, not the --level ladder**
   Files: `docs/architecture/artifact-compiler.md`, `docs/contributing/generated-code.md`, `docs/reference/bundle-manifest.md`
9. **F — Generated-program and launch facts are stale**
   Files: `docs/architecture/code-map.md`, `docs/architecture/execution-orchestrator.md`
10. **H — Two pilot-metrics finding codes are missing from the error inventory**
   Files: `docs/reference/error-codes.md`
11. **I — Execution-disabled mode is undocumented**
   Files: `docs/reference/api.md`
12. **D — The MoE slice was never swept into inventories, navigation, and registry tables**
   Files: `docs/architecture/code-map.md`, `docs/contributing/changing-contracts.md`, `docs/index.md`, `docs/reference/method-registry.md`

---

## Root causes

### A. MLX-LM is exempt from the device four-bit capability gate

planning.py:549 excludes TrainingRuntime.MLX_LM from the four-bit device gate. Eligibility comes from four-bit quantization metadata in the pinned model revision, checked at model-data validation. The discovered Apple device is built with supports_4bit=False (profiling.py:913), so any doc requiring a device four-bit fact tells Apple Silicon users a supported path is unsupported.

### B. MLX unified-memory admission is now a live gate at model-data time

_bundle_programs/mlx/validate.py:174 runs require_unified_memory_admission BEFORE any weights load. It measures the packed safetensors shards, adds packed-size excess over planned resident bytes to the point and upper estimates, and refuses unless available >= max(adjusted point, adjusted upper) + max(user reserve, 8 GiB). Contract bumped to aptus.mlx-unified-memory-admission.v2. This produced the recorded 2026-07-28 Qwen3 30B refusal with an 18.932 GiB shortfall.

### L. MLX-LM refuses whole-text rows, and no document says so

generation.py:216 raises for any row carrying a string `text` field, because pinned MLX-LM 0.31.3 cannot combine full-text supervision with the bundle's required prompt masking. Grep-confirmed: no doc in the repository states this.

### J. --reserve-gib is silently floored to 8.0 GiB under mps

cli.py:547 raises the reserve to at least 8.0 GiB whenever --backend mps is selected; api.py:417 applies the same floor to declared facts. The documented 2.0 default is wrong for every Apple Silicon plan.

### C. model-data-evidence.json is now a tolerated mutable bundle path

plan_contract.py:192 added it to mutable_files. Docs that enumerate the mutable set as closed would tell an operator a valid bundle is corrupt.

### G. MLX quantized storage is priced from the bound quantization_layout

plan_contract.py:902 prices from the layout rather than a flat 0.5P/0.0625P, so 8-bit router-gate overrides cost strictly more than 0.5P. aptus-memory-mlx-v2 was bumped; the formula doc was not.

### K. The expert routing rail drops evidence on the runtime-mismatch branch

ExpertTopologyRail.tsx:116 omits method, placement, and the pilot-boundary reason whenever the conditional path's supported runtime differs from the selected target runtime — which is the default state, since a fresh draft starts on the CUDA backend (demo.ts:46). The branch is untested.

### E. MLX preflight.py is a 23-line platform check, not the --level ladder

The cumulative six-level orchestrator exists only in the CUDA preflight.py. For MLX the ladder lives in validate.py, which merely spawns preflight.py as a subprocess.

### F. Generated-program and launch facts are stale

No TRAIN_SCRIPT/RUN_SCRIPT/PREFLIGHT_SCRIPT/VALIDATE_SCRIPT constants exist in generation.py; the programs are package data read via importlib.resources, MLX emits five, and single-device CUDA launches train.py directly rather than through Accelerate.

### H. Two pilot-metrics finding codes are missing from the error inventory

validation.py:1602 emits PILOT_METRICS_INVALID and PILOT_METRICS_UNBOUND at severity error.

### I. Execution-disabled mode is undocumented

api.py:1592 returns 403 desktop_execution_disabled from POST /api/v1/jobs and from POST /api/v1/validate with run: true. api.md never mentions the mode.

### D. The MoE slice was never swept into inventories, navigation, and registry tables

e97ef5e added six versioned contracts, a fifth adapter-target family (qwen3_moe), and an evidence record, none of which reached the contract inventory, the docs index, or the method registry reference.

---

## Findings

### High severity (12)

#### `docs/reference/capability-matrix.md:47` — DRIFTED (root cause A)

The capability matrix says the MLX-LM QLoRA row needs explicit four-bit hardware capability facts, but the planner deliberately exempts MLX-LM from the four-bit device-capability gate and admits QLoRA on an MPS device whose supports_4bit flag is false.

- **Doc says** (`docs/reference/capability-matrix.md:47`): | QLoRA | Conditional through uninterrupted pilot and full-duration adapter training, with explicit four-bit capability facts and MLX model metadata | Unsupported | Unsupported | MLX-LM adapter |
- **Code does** (`src/aptus/planning.py:549`): and runtime_contract.training_runtime != TrainingRuntime.MLX_LM
- **Fix**: Replace "with explicit four-bit capability facts and MLX model metadata" with wording that names only the model-side gate, e.g. "with explicit four-bit quantization metadata in the pinned MLX model revision; no device four-bit capability fact is required". This also removes the internal contradiction with this document's own rows at line 80 ("Explicit quantization metadata in the pinned MLX model") and line 106 ("It does not infer BF16, four-bit, eight-bit, or free VRAM").

#### `docs/reference/validation-states.md:132` — MISSING (root cause B)

The MLX branch of the model-data level omits the live unified-memory admission gate: model-data now measures the packed checkpoint and refuses before any weights load when available unified memory is below the adjusted upper estimate plus an 8 GiB reserve.

- **Doc says** (`docs/reference/validation-states.md:132`): For `mlx-lm`, model-data validation loads the exact pinned revision through
- **Code does** (`src/aptus/_bundle_programs/mlx/validate.py:174`): admission = require_unified_memory_admission(plan, model_path)
- **Fix**: Add a sentence to the `mlx-lm` model-data paragraph: before calling the MLX-LM loader, model-data validation scans the pinned safetensors shards, adds any packed-size excess over the planned resident bytes to the point and upper estimates, and requires current available unified memory to be at least max(adjusted point, adjusted upper) plus max(plan reserve, 8 GiB); otherwise it fails closed with the exact required/available/shortfall byte counts and records `aptus.mlx-unified-memory-admission.v2` in `model-data-evidence.json`. Cite docs/operations/evidence/2026-07-28-qwen3-moe-admission/README.md as the recorded instance of this refusal.

#### `docs/reference/bundle-manifest.md:175` — DRIFTED (root cause C)

bundle-manifest.md's normative "Mutable runtime paths" list omits `model-data-evidence.json`, which the manifest validator now tolerates, so the doc's closed enumeration and its "any other unmanifested file invalidates the bundle" rule are both wrong for every MLX bundle that has passed the model-data gate.

- **Doc says** (`docs/reference/bundle-manifest.md:175`): The manifest permits only these unmanifested files and prefixes:

```text
.validation-report.lock
validation-report.json
preflight-metrics.json
pilot-output/
runs/
```

Any other unmanifested file invalidates the bundle.
- **Code does** (`src/aptus/plan_contract.py:192`): mutable_files = {
        ".validation-report.lock",
        "model-data-evidence.json",
        "validation-report.json",
        "preflight-metrics.json",
    }
    mutable_prefixes = ("pilot-output/", "runs/")
- **Fix**: Add `model-data-evidence.json` to the fenced mutable-path list at bundle-manifest.md:178-182 (matching plan_contract.py's set order), and add a short note under "Measured preflight output" (or a new "Model-data evidence" subsection near line 190) explaining that the MLX model-data gate writes `aptus.mlx-model-data-evidence.v1` into the bundle root and that validation-report `bindings.model_data_evidence` binds its SHA-256 (src/aptus/_bundle_programs/mlx/validate.py:242 and :91). The ZIP-exclusion list at lines 309-315 does not need it, because the archive is written at compile time before model-data runs.

#### `docs/methodology/memory-estimation.md:288` — DRIFTED (root cause B)

memory-estimation.md's MLX fit ladder still describes the pre-MoE admission sequence: it names admission only at pilot and train, and omits both the model-data-time gate and the new packed-checkpoint resident adjustment that e97ef5e added when it bumped aptus.mlx-unified-memory-admission v1 to v2.

- **Doc says** (`docs/methodology/memory-estimation.md:288`): prove pilot or full-run fit. The MLX pilot runs uninterrupted from the pinned
base and rechecks live unified-memory admission. Train admission later requires
current available memory above the measured pilot peak plus reserve.
- **Code does** (`src/aptus/_bundle_programs/mlx/validate.py:174`): admission = require_unified_memory_admission(plan, model_path)
    model, tokenizer, config = load(
- **Fix**: In "Device budget and fit", add the model-data-time admission step before the pilot step, and state its arithmetic: the gate measures the pinned snapshot's safetensors bytes, computes adjustment = max(0, observed_safetensors_bytes - (W+Q)), and refuses unless live available unified memory >= max(M_point+adjustment, M_upper+adjustment) + max(user reserve, 8 GiB). Note that this reserve floor is a hard 8 GiB minimum independent of the planner's user reserve, and that the refusal happens before the model is loaded.

#### `docs/methodology/memory-estimation.md:123` — DRIFTED (root cause G)

The normative MLX QLoRA base-weight and quantization-metadata coefficients are stated as flat 0.5P / 0.0625P, but aptus-memory-mlx-v2 now prices quantized storage from the bound quantization_layout, so a model with 8-bit router-gate overrides costs strictly more than 0.5P.

- **Doc says** (`docs/methodology/memory-estimation.md:123`): base weights use $0.5P$ bytes and quantization metadata uses $0.0625P$ bytes.
- **Code does** (`src/aptus/plan_contract.py:902`): storage_bytes = round(
        (default_parameters * default_bits + weighted_storage_bits) / 8
    )
    metadata_bytes = round(
        default_parameters * 4 / default_group_size + weighted_metadata_bytes
    )
- **Fix**: Split the QLoRA paragraph into the two branches the code actually has. Keep 0.5P / 0.0625P as the layout-less fallback (plan_contract.py:850), and add the layout-driven rule: W = (P_default * default_bits + sum over overrides of P_o * bits_o) / 8 and Q = P_default * 4 / default_group_size + sum over overrides of P_o * 4 / group_size_o, where each router-gate override covers hidden_size * expert_count parameters (plan_contract.py:894). State that the reviewed Qwen3 MoE layout is four-bit group-64 by default with one eight-bit group-64 override per layer, matching the assumption string planning.py:235 already emits.

#### `docs/reference/method-registry.md:121` — DRIFTED (root cause A)

method-registry.md's QLoRA gate list asserts that every participating device must explicitly support the four-bit path, but planning.py exempts the MLX-LM runtime from the device four-bit gate entirely.

- **Doc says** (`docs/reference/method-registry.md:121`): - every participating device must explicitly support the four-bit path;
- **Code does** (`src/aptus/planning.py:549`): and runtime_contract.training_runtime != TrainingRuntime.MLX_LM
        and (
            not participating_devices
            or any(not item.supports_4bit for item in participating_devices)
        )
- **Fix**: Scope the bullet to CUDA: "every participating CUDA device must explicitly support the four-bit path", and add an MLX-LM bullet stating that eligibility comes from the pinned revision's four-bit MLX quantization metadata at model-data validation, not a device capability bit. While editing this section, also scope lines 123-124 (reentrant vs non-reentrant gradient checkpointing), which planning.py:648-659 applies only to transformers-peft-cuda.

#### `docs/operations/operator-checklist.md:177` — MISSING (root cause B)

The operator checklist's Model-data action section never mentions the live unified-memory admission gate that MLX model-data validation enforces before the model loads, so an operator following the five-ordered-action procedure has no way to anticipate or interpret the exact failure that blocked the 2026-07-28 Qwen3 MoE attempt.

- **Doc says** (`docs/operations/operator-checklist.md:177`): Model-data validation does not enter training mode and does not prove
accelerator fit. MLX QLoRA must obtain four-bit eligibility from the pinned
model metadata, not a CUDA-style device flag.
- **Code does** (`src/aptus/_bundle_programs/mlx/validate.py:174`): admission = require_unified_memory_admission(plan, model_path)
    model, tokenizer, config = load(
        str(model_path),
- **Fix**: Add a checkbox to the Model-data action list (lines 167-175) stating that for MLX-LM the action measures the packed safetensors shards, compares live available unified memory against the packed-checkpoint-adjusted candidate estimate plus the 8 GiB reserve, and refuses before any weight load if the shortfall is positive; and note that a passing action writes an immutable `model-data-evidence.json` binding `aptus.mlx-model-data-evidence.v1`. Also extend the ordered-actions table row 2 (line 137) 'Inspect before continuing' column with the admission record.

#### `docs/architecture/code-map.md:94` — DRIFTED (root cause F)

code-map.md's "generated-runtime boundary" section says the compiler emits four programs from `TRAIN_SCRIPT`/`RUN_SCRIPT`/`PREFLIGHT_SCRIPT`/`VALIDATE_SCRIPT` constants in generation.py; no such constants exist anywhere in the repository, the programs are package-data files read through importlib.resources, and the MLX runtime emits five programs.

- **Doc says** (`docs/architecture/code-map.md:94`): The compiler emits four executable programs from runtime-specific constants in
[`generation.py`](../../../../src/aptus/generation.py):

- `TRAIN_SCRIPT` becomes `train.py`;
- `RUN_SCRIPT` becomes `run.py`;
- `PREFLIGHT_SCRIPT` becomes `preflight.py`;
- `VALIDATE_SCRIPT` becomes `validate.py`.
- **Code does** (`src/aptus/generation.py:28`): _BUNDLE_PROGRAMS = {
    "cuda": ("train.py", "run.py", "preflight.py", "validate.py"),
    "mlx": ("train.py", "run.py", "reload.py", "preflight.py", "validate.py"),
}
...
line 40:    resource = resources.files("aptus").joinpath("_bundle_programs", runtime, name)
line 812:        (root / name).write_bytes(_bundle_program_bytes(program_runtime, name))

`grep -rn "TRAIN_SCRIPT|RUN_SCRIPT|PREFLIGHT_
- **Fix**: Replace the constant list with the real mechanism: the CUDA compiler emits four programs and the MLX compiler emits five (adding `reload.py`), enumerated by `_BUNDLE_PROGRAMS` in generation.py and copied byte-for-byte from the package resources under `src/aptus/_bundle_programs/{cuda,mlx}/` via `importlib.resources` (`_bundle_program_bytes`, generation.py:34-47). system.md:101-104 already states this correctly and can be mirrored. The paragraph's own later sentence about "fresh-reload sources" already contradicts the "four executable programs" count.

#### `docs/guides/compare-plans.md:23` — DRIFTED (root cause A)

compare-plans.md states flatly that CUDA is required, but the method registry binds LoRA and QLoRA to the mps backend via the mlx-lm runtime, so single-device Apple Silicon candidates are enumerated and rankable.

- **Doc says** (`docs/guides/compare-plans.md:23`): - CUDA is required. Full fine-tuning requires BF16. Adapter methods select BF16 when declared and can select FP16 otherwise, subject to the exact pilot.
- **Code does** (`src/aptus/methods/registry.py:123`): supported_backends=("cuda", "mps"),  # qlora descriptor; the lora descriptor carries the same tuple at registry.py:58, and each has an mlx-lm/mps runtime_binding with supported_distributions=("single",)
- **Harm**: An Apple Silicon owner reading the plan-comparison guide concludes no candidate in the matrix can ever execute on their machine and abandons Aptus (or buys CUDA hardware) for a LoRA/QLoRA job that the mlx-lm compiler actually supports single-device today.
- **Fix**: Change the bullet to scope the CUDA claim: "CUDA is required for full and int8-LoRA. LoRA and QLoRA also compile on one Apple unified-memory device through the mlx-lm runtime (single placement only)." Then add an MLX row to the support-rules list rather than leaving the section CUDA-exclusive.

#### `docs/guides/choose-a-method.md:19` — DRIFTED (root cause A)

choose-a-method.md asserts that every row of the selectable-method table needs a CUDA backend, contradicting the mlx-lm/mps runtime bindings for LoRA and QLoRA.

- **Doc says** (`docs/guides/choose-a-method.md:19`): Every row still needs a CUDA backend, a supported model family, exact batch arithmetic, resource checks, model-data validation, measured preflight, and the selected real-model pilot.
- **Code does** (`src/aptus/methods/registry.py:58`): supported_backends=("cuda", "mps"),  # lora descriptor; runtime_bindings include _runtime(training_runtime="mlx-lm", compute_backend="mps", compiler_id="mlx-lm.lora.v1", supported_distributions=("single",))
- **Harm**: A user on a Mac reads the one guide named "Choose a Fine-Tuning Method", sees that all four methods demand CUDA, and never selects the MLX-LM LoRA/QLoRA path that Aptus compiles, validates, pilots, and full-duration trains on their machine — the exact wrong decision this guide exists to prevent.
- **Fix**: Replace "a CUDA backend" with "a supported backend (CUDA for full and int8-LoRA; CUDA or one Apple unified-memory device for LoRA and QLoRA)", and add to the "Choose QLoRA" section (line 58) that the CUDA four-bit device-capability requirement does not apply to MLX-LM, where eligibility comes from explicit four-bit quantization metadata in the pinned model revision. Also note the reviewed qwen3_moe slice is executable only as single-device MLX-LM QLoRA.

#### `docs/guides/prepare-a-dataset.md:37` — MISSING (root cause L)

prepare-a-dataset.md presents whole-text rows (and therefore every .txt source) as a first-class accepted schema without disclosing that MLX-LM compilation refuses them outright.

- **Doc says** (`docs/guides/prepare-a-dataset.md:37`): Whole-text rows train on every retained token.
- **Code does** (`src/aptus/generation.py:216`): "MLX-LM compilation refuses text rows because pinned MLX-LM 0.31.3 "
            "cannot combine full-text supervision with the bundle's required prompt masking."  # raised from _mlx_training_row (generation.py:210) for any row carrying a string `text` field
- **Harm**: An Apple Silicon user curates and reviews a whole-text corpus — or any `.txt` file, which the container table at line 18 converts to `{"text": ...}` rows — then discovers at compile time that the bundle cannot be produced at all, wasting the entire data-preparation and human-review pass this guide prescribes. No other document in docs/ states this restriction (grep for whole-text/text row/refuses text returns only this file).
- **Fix**: Add a sentence to the "Whole-text supervision" subsection: "MLX-LM compilation refuses whole-text rows because pinned MLX-LM 0.31.3 cannot combine full-text supervision with the bundle's required prompt masking. For an Apple Silicon target, use prompt/completion, instruction/output, or messages rows." Flag the `.txt` container row in the table the same way.

#### `docs/guides/troubleshooting.md:55` — MISSING (root cause B)

troubleshooting.md has no entry for the MLX live unified-memory admission refusal, the one hard failure that actually blocked the recorded 2026-07-28 Qwen3 run, and its "Model-data validation fails" checklist names only model/data/credential causes.

- **Doc says** (`docs/guides/troubleshooting.md:55`): Confirm network or cache access, repository ID, immutable revision, tokenizer, model family, parameter count, gated-model credentials, and every canonical training row.
- **Code does** (`src/aptus/_bundle_programs/mlx/validate.py:174`): require_model_data() calls `admission = require_unified_memory_admission(plan, model_path)` at line 174, BEFORE `model, tokenizer, config = load(...)` at line 175. train.py:619-625 raises "Current available Apple unified memory is below the packed-checkpoint-adjusted candidate upper estimate plus the required 8 GiB Aptus reserve. required=... available=... shortfall=... bytes." No section of troub
- **Harm**: An Apple Silicon operator whose `--action model-data` job dies with a byte-denominated unified-memory shortfall opens the guide's "Model-data validation fails" section and is sent to verify network access, repository ID, revision, tokenizer, gated-model credentials and dataset rows — none of which is the cause. They will re-pull the model, re-check HF tokens, or replan the dataset instead of freeing memory or lowering the candidate, and the "Preflight or pilot runs out of memory" heading actively tells them memory refusals only happen two levels later.
- **Fix**: Add a "## MLX model-data refuses before the model loads" section: state that on MLX-LM the model-data level measures the packed safetensors shards and live available unified memory before any weight loads, adds any packed-size excess over planned resident bytes to both the point and upper estimates, and refuses unless available >= max(adjusted point, adjusted upper) + max(user reserve, 8 GiB); tell the operator to read `required=`/`available=`/`shortfall=` from the job log, free memory by normal reboot or application shutdown and re-run model-data, or replan a smaller candidate; link the recorded refusal at docs/operations/evidence/2026-07-28-qwen3-moe-admission/README.md. Also rename the line-62 heading so it no longer implies memory refusals begin at preflight.

### Medium severity (17)

#### `docs/architecture/data-and-identity-flow.md:148` — DRIFTED (root cause C)

The doc enumerates the exact set of mutable paths tolerated outside the compiler-managed manifest, but the manifest validator now also tolerates `model-data-evidence.json`, which the list omits.

- **Doc says** (`docs/architecture/data-and-identity-flow.md:148`): These mutable paths are intentionally outside the compiler file list:
- **Code does** (`src/aptus/plan_contract.py:192`): "model-data-evidence.json",
- **Fix**: Add `- `model-data-evidence.json`;` to the bullet list at docs/architecture/data-and-identity-flow.md lines 150-154 (it is the MLX `aptus.mlx-model-data-evidence.v1` artifact written by the generated MLX validator and bound by digest in the validation report), so the enumerated integrity boundary matches `mutable_files` in `validate_bundle_manifest`.

#### `docs/architecture/artifact-compiler.md:61` — DRIFTED (root cause E)

The doc describes `preflight.py` as the cumulative multi-level validation orchestrator, but that is only true of CUDA bundles; the MLX bundle's `preflight.py` accepts no `--level` at all and only checks platform plus pinned dependency versions, with `validate.py` doing the level orchestration.

- **Doc says** (`docs/architecture/artifact-compiler.md:61`): - `preflight.py`: cumulative runtime-validation orchestrator for dependency,
- **Code does** (`src/aptus/_bundle_programs/mlx/preflight.py:2`): """Fail-closed MLX-LM dependency and uninterrupted-run preflight."""
- **Fix**: Split the bullet by runtime the way the neighbouring `train.py` and `config/*.yaml` bullets already do: state that the CUDA `preflight.py` is the cumulative `--level` orchestrator (`LEVELS` / `target >= LEVELS.index(...)` in `_bundle_programs/cuda/preflight.py`), while the MLX `preflight.py` is an argument-free Apple-silicon and pinned-dependency gate that `validate.py` shells out to, with the MLX level sequencing living in `validate.py:main`.

#### `docs/contributing/changing-contracts.md:33` — MISSING (root cause D)

The contract inventory table is not updated with the six versioned cross-boundary contracts introduced by the Qwen3 MoE slice, so a contributor consulting the normative inventory cannot find the contracts they must version-bump.

- **Doc says** (`docs/contributing/changing-contracts.md:33`): | MLX artifact manifest | `aptus.mlx-artifact-manifest.v1` | generated MLX action owner and parent verifier |
- **Code does** (`src/aptus/plan_contract.py:569`): "schema_version": "aptus.model-architecture-contract.v1",
- **Fix**: Add rows for the contracts introduced by e97ef5e: `aptus.model-architecture-contract.v1` (`plan_contract.py`), `aptus.mlx-model-load-binding.v3`, `aptus.mlx-model-parameter-census.v1`, `aptus.mlx-packed-checkpoint.v1`, `aptus.mlx-unified-memory-admission.v2`, and `aptus.mlx-model-data-evidence.v1` (generated MLX runtime and parent verifier). Every other MLX runtime-evidence contract of the same kind is already listed, so their absence reads as "not a versioned contract".

#### `docs/product/current-capabilities.md:70` — DRIFTED (root cause A)

The product boundary page repeats the claim that MLX-LM QLoRA requires explicit four-bit capability facts, but MPS single-device compatibility accepts LoRA and QLoRA without consulting any device capability flag.

- **Doc says** (`docs/product/current-capabilities.md:70`): LoRA and QLoRA. MLX-LM QLoRA requires explicit four-bit capability facts and
- **Code does** (`src/aptus/planning.py:314`): return method in {Method.LORA, Method.QLORA}
- **Fix**: Change to "MLX-LM QLoRA requires four-bit quantization metadata in the pinned MLX model revision, verified at model-data validation rather than from a device capability flag." The discovered Apple compatibility device is created with supports_4bit=False (src/aptus/profiling.py:913), so the current wording would make every discovered MLX QLoRA plan ineligible, contradicting the 2026-07-27 acceptance record.

#### `docs/index.md:118` — MISSING (root cause D)

docs/index.md — the documentation navigation hub — never mentions the shipped Qwen3 MoE compatibility slice or its dated 2026-07-28 admission evidence record, even though it enumerates both 2026-07-27 evidence records twice and README, release-gates.md, apple-silicon-pilot.md, and current-capabilities.md were all updated for it.

- **Doc says** (`docs/index.md:118`): - [2026-07-27 desktop engineering acceptance](../2026-07-27-desktop-release/README.md)
- **Code does** (`docs/operations/evidence/2026-07-28-qwen3-moe-admission/README.md:1`): # Qwen3 30B-A3B MoE admission and performance evidence
- **Fix**: In docs/index.md add `- [2026-07-28 Qwen3 MoE admission and performance evidence](../2026-07-28-qwen3-moe-admission/README.md)` after line 118, add a matching "Inspect the Qwen3 MoE admission attempt" row to the "Choose by goal" table beside the two existing evidence rows (lines 25-26), and extend the "Evidence notice" (lines 169-176) with one sentence stating that the exact Qwen3 MoE MLX-LM QLoRA row is conditional and has only safe-refusal evidence (the recorded 30B attempt stopped before model loading), matching README.md lines 39-47. Do not change any code.

#### `docs/reference/error-codes.md:165` — MISSING (root cause H)

error-codes.md's "Runtime invocation and attestation" table — the normative inventory of host validator findings — omits the two MLX pilot-metrics finding codes `PILOT_METRICS_INVALID` and `PILOT_METRICS_UNBOUND`, both of which validate_bundle emits at severity error.

- **Doc says** (`docs/reference/error-codes.md:165`): | `PREFLIGHT_METRICS_INVALID` | error | Runtime-specific measured-preflight metrics are missing, malformed, non-positive, or misbound |
| `PREFLIGHT_METRICS_UNBOUND` | error | The report digest or embedded metrics do not match the measured file |
- **Code does** (`src/aptus/validation.py:1602`): findings.append(
                        _finding(
                            "PILOT_METRICS_INVALID",
                            str(error),
                            path="pilot-output/metrics.json",
                        )
                    )
                    runtime_attestation_valid = False
... (line 1617)
                            _finding(
                                "PILOT
- **Fix**: Append two rows after error-codes.md:165: "| `PILOT_METRICS_INVALID` | error | MLX `pilot-output/metrics.json` is missing, malformed, or does not bind the plan, candidate, uninterrupted action, or measured evidence |" and "| `PILOT_METRICS_UNBOUND` | error | The runtime validation report does not bind the exact MLX pilot metrics digest |". Neither string appears anywhere under docs/, so no other document owns them.

#### `docs/reference/bundle-manifest.md:133` — DRIFTED (root cause E)

bundle-manifest.md describes `preflight.py` as the cumulative six-level `--level` orchestrator invoked by `validate.py`, which is true only for the CUDA bundle; the MLX `preflight.py` is a 23-line dependency/platform check with no arguments, and MLX's `validate.py` owns the level ladder itself.

- **Doc says** (`docs/reference/bundle-manifest.md:133`): Despite its filename, `preflight.py` is not only the synthetic check. It is a
cumulative level orchestrator:

1. contract validation;
2. Python static parsing;
3. exact direct dependency checks;
4. `train.py --preflight-model-data`;
5. runtime-specific measured preflight; and
6. runtime-specific pilot.
- **Code does** (`src/aptus/_bundle_programs/mlx/validate.py:844`): completed = subprocess.run([sys.executable, str(ROOT / "preflight.py")], cwd=ROOT)

(mlx/validate.py:825-826 owns `--level`; mlx/preflight.py is 23 lines with no argparse and only checks Darwin/arm64 plus the mlx==0.31.2 / mlx-lm==0.31.3 pins. By contrast cuda/preflight.py:32 defines LEVELS and :434 defines `--level`, and cuda/validate.py:638-641 acquires the lease and invokes preflight.py with th
- **Fix**: Split the `preflight.py` section (bundle-manifest.md:131-147) by runtime: keep the six-level `--level` orchestrator description for CUDA, and state that MLX `preflight.py` is only the Apple-silicon and pinned-dependency check, while MLX `validate.py` owns `--level` and runs model-data, `run.py --bounded-smoke`, and `run.py --pilot` directly. Also correct the file-purpose row at line 116 (`Cumulative level executor used by validate.py`) and the `validate.py` sentence at lines 125-126 (`It acquires the portable lease, invokes preflight.py at the requested level`) — MLX `validate.py` imports no lease at all and invokes `preflight.py` with no arguments.

#### `docs/reference/method-registry.md:67` — DRIFTED (root cause D)

The selectable method matrix transcribes descriptor fields but two cells no longer match registry.py: qlora's base_storage is "runtime-native-four-bit" (not bitsandbytes NF4), and both lora and qlora declare supported_backends ("cuda", "mps"), not CUDA alone.

- **Doc says** (`docs/reference/method-registry.md:67`): | `qlora` | Frozen base plus adapter | Bitsandbytes NF4 with double quantization | `transformers.peft-qlora.v2` | `peft-adapter-safetensors` | CUDA | `single`, `ddp` |
- **Code does** (`src/aptus/methods/registry.py:120`): base_storage="runtime-native-four-bit",
            compiler_id="transformers.peft-qlora.v2",
            export_kind="peft-adapter-safetensors",
            supported_backends=("cuda", "mps"),
- **Fix**: Set the qlora Base storage cell to "Runtime-native four-bit" and the Backend cell for both lora (line 65) and qlora to "CUDA, MPS". Either add per-runtime rows for the mlx-lm.lora.v1 / mlx-lm.qlora.v1 compilers and their mlx-lm-adapter export, or add a note that the Compiler and Export columns show only the CUDA binding and that runtime_bindings carries the MLX-LM pair. Also correct line 117, "QLoRA trains LoRA adapters through a frozen NF4 double-quantized base", which contradicts registry.py:115's "frozen runtime-native four-bit base".

#### `docs/reference/method-registry.md:89` — DRIFTED (root cause D)

method-registry.md's LoRA section enumerates the adapter target catalog as four families, but catalog.py registers a fifth, qwen3_moe, with an attention-only target tuple.

- **Doc says** (`docs/reference/method-registry.md:89`): modules. The current target catalog covers `llama`, `mistral`, `gemma`, and
`qwen` projection names.
- **Code does** (`src/aptus/catalog.py:50`): TARGET_MODULES[QWEN3_MOE_FAMILY] = QWEN3_MOE_TARGET_MODULES
- **Fix**: Extend the sentence to five families and state that `qwen3_moe` resolves to attention-only projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), matching catalog.py:24 and docs/reference/capability-matrix.md:160. The "Planner behavior" filter list at lines 219-228 has the same gap and should gain a MoE identity, quantization-layout, and topology bullet.

#### `docs/methodology/preflight-calibration.md:25` — MISSING (root cause B)

The methodology doc that owns model-data validation describes the MLX-LM gate as only verifying load and tokenization compatibility, omitting the live unified-memory admission and architecture-contract checks that now run before the model is loaded and can refuse the gate outright.

- **Doc says** (`docs/methodology/preflight-calibration.md:25`): MLX-LM gate verifies load and tokenization compatibility without an optimizer
step.
- **Code does** (`src/aptus/_bundle_programs/mlx/validate.py:174`): architecture_contract = require_method_model(plan, candidate, model_path)
    admission = require_unified_memory_admission(plan, model_path)
    model, tokenizer, config = load(
- **Fix**: Add to the "Model-data validation" section that the MLX-LM gate first binds the pinned architecture contract (model type, architecture, expert topology, canonical quantization layout), then measures the packed checkpoint and enforces live unified-memory admission before any model load, and finally seals `model-data-evidence.json` under `aptus.mlx-model-data-evidence.v1`. The current wording implies a memory-neutral compatibility probe, but this is the gate that refused the 2026-07-28 Qwen3 MoE run with an 18.932 GiB shortfall (docs/operations/evidence/2026-07-28-qwen3-moe-admission/README.md:16).

#### `docs/reference/method-registry.md:243` — MISSING (root cause D)

The export-kind contract table lists only the two CUDA export kinds; `mlx-lm-adapter`, which the compiler writes into config/trainer.json for every MLX-LM candidate, is absent here and from every other document.

- **Doc says** (`docs/reference/method-registry.md:243`): | `peft-adapter-safetensors` | Adapter weights and reloadable PEFT adapter metadata over the pinned base revision |
- **Code does** (`src/aptus/methods/registry.py:76`): export_kind="mlx-lm-adapter",
- **Fix**: Add a third row: `mlx-lm-adapter` | MLX-native adapter weights and adapter configuration reloadable on the pinned base revision. plan_contract.py:75 and :93 accept it as a valid export kind and generation.py:120 writes the runtime binding's export_kind, so a reader of this table cannot currently interpret an MLX bundle's trainer.json. A grep of docs/ finds zero occurrences of the string.

#### `docs/product/ui-ux.md:85` — DRIFTED (root cause K)

ui-ux.md's normative sentence says the expert routing rail displays the compatibility result's runtime, method, placement, and pilot boundary, but ExpertTopologyRail drops method, placement, and the pilot-boundary reason whenever the conditional path's supported runtime differs from the selected target runtime. Failure scenario: a fresh draft defaults to device backend "cuda" and target runtime "transformers-peft-cuda" (web/src/demo.ts:46 and web/src/demo.ts:65). An operator who inspects the reviewed Qwen3 MoE checkpoint before switching the backend to mps gets compatibility {status: "conditional", supported_runtime: "mlx-lm", supported_methods: ["qlora"], distribution: "single", reason: "...Measured preflight and a real-model pilot remain mandatory."} from src/aptus/inspection.py:365-378. ExpertTopologyRail.tsx:34 then computes runtimeMatches=false, so the ternary at lines 114-116 renders only the mismatch sentence: the method (qlora), the placement (single), and the mandatory-preflight-and-pilot boundary are never shown. Nothing else in the workbench renders the field — `compatibility` is consumed only by ExpertTopologyRail (App.tsx:464, FactsStage.tsx:310-312). The doc sentence at lines 84-85 is unconditional, so it promises four pieces of evidence that this reachable (in fact default) state omits three of. The mismatch branch is also untested: ExpertTopologyRail.test.tsx:31 and FactsStage.test.tsx:97 both exercise selectedRuntime="mlx-lm" only.

- **Doc says** (`docs/product/ui-ux.md:85`): runtime, method, placement, and pilot boundary from the compatibility result.
- **Code does** (`web/src/components/ExpertTopologyRail.tsx:116`): `The conditional path requires ${supportedRuntime}. The current ${selectedRuntime} target remains unsupported for this model.`
- **Fix**: Pick one side. Either keep the compatibility evidence in the runtime-mismatch branch (append the supported method, placement, and compatibility.reason to the "The conditional path requires ..." sentence at ExpertTopologyRail.tsx:116), or narrow ui-ux.md:84-85 to state that the rail displays the exact runtime, method, placement, and pilot boundary when the compatibility result binds the selected target runtime, and otherwise names the required runtime and marks the current target unsupported. Add a test for the runtime-mismatch presentation either way.

#### `docs/contributing/generated-code.md:25` — DRIFTED (root cause E)

The generated-sources table says `_bundle_programs/mlx/preflight.py` performs 'MLX cumulative runtime action orchestration', but that file is a 23-line Apple-silicon platform and pinned-version gate that takes no arguments; the cumulative level ladder lives in `mlx/validate.py`, which merely spawns preflight.py as a subprocess.

- **Doc says** (`docs/contributing/generated-code.md:25`): | `_bundle_programs/mlx/preflight.py` | `preflight.py` | MLX cumulative runtime action orchestration |
- **Code does** (`src/aptus/_bundle_programs/mlx/preflight.py:10`): def main() -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX-LM requires Apple silicon macOS.")
    expected = {"mlx": "0.31.2", "mlx-lm": "0.31.3"}
- **Fix**: Change the row's Responsibility to something like 'Apple-silicon platform and exact mlx/mlx-lm pin gate; no level argument', and note in the mlx/validate.py row (line 26) that it owns the cumulative `--level` ladder and invokes preflight.py as the dependency step. This is the same defect round 1 confirmed at docs/architecture/artifact-compiler.md:61 (preflight.py described as the cumulative --level orchestrator, true only for CUDA), but the fix location is this contributor table.

#### `docs/operations/state-storage-retention.md:72` — MISSING (root cause C)

The bundle mutability boundary enumerates the allowed mutable roots but omits `model-data-evidence.json`, which the MLX model-data validator now writes into the bundle root and which the manifest checker explicitly tolerates, so the documented list would classify a legitimate runtime artifact as an unexpected file.

- **Doc says** (`docs/operations/state-storage-retention.md:72`): - `.validation-report.lock`;
- `validation-report.json`;
- `preflight-metrics.json`;
- `pilot-output/`; and
- `runs/`.
- **Code does** (`src/aptus/plan_contract.py:190`): mutable_files = {
        ".validation-report.lock",
        "model-data-evidence.json",
        "validation-report.json",
        "preflight-metrics.json",
    }
- **Fix**: Add `model-data-evidence.json` to the allowed mutable roots list, describing it as the MLX model-data evidence record (`aptus.mlx-model-data-evidence.v1`) that binds plan, candidate, model revision, model-load binding, and unified-memory admission. Same root cause as the round-1 finding on docs/architecture/data-and-identity-flow.md:148, but a separate list in a separate operator-facing doc; note that no doc in docs/ currently mentions this filename at all.

#### `docs/reference/cli.md:81` — DRIFTED (root cause J)

The CLI reference documents `--reserve-gib` as a plain non-negative reserve defaulting to 2.0, but the CLI silently raises the reserve to at least 8.0 GiB whenever `--backend mps` is selected, so the documented contract is wrong for every Apple Silicon plan and the planner's feasibility outcome differs from the documented input.

- **Doc says** (`docs/reference/cli.md:81`): | `--reserve-gib NUMBER` | No | `2.0` | Non-negative reserve subtracted from each device |
- **Code does** (`src/aptus/cli.py:547`): if backend == Backend.MPS:
        reserve_gib = max(reserve_gib, 8.0)
- **Fix**: Amend the Contract cell to state that the supplied value is raised to a minimum of 8.0 GiB when `--backend mps` is planned, matching the CLI's own help string ('default: 2; Apple unified memory minimum: 8'). docs/reference/configuration-defaults.md:141 documents an 8 GiB floor only for the API local-scan path, so no other doc covers the CLI behavior.

#### `docs/architecture/code-map.md:42` — DRIFTED (root cause D)

The module-responsibility table describes plan_contract.py as owning only "Canonical candidate/plan identities and bundle-manifest verification", but it is now also the single source of the portable MLX unified-memory formula and of the model-architecture / quantization-layout contract that the MLX bundle programs and train admission both enforce.

- **Doc says** (`docs/architecture/code-map.md:42`): | [`plan_contract.py`](../../../../src/aptus/plan_contract.py) | Canonical candidate/plan identities and bundle-manifest verification | Runtime artifact success |
- **Code does** (`src/aptus/plan_contract.py:911`): line 12:  MLX_FORMULA_VERSION = "aptus-memory-mlx-v2"
line 493: def expected_model_architecture_contract(
line 590: def validate_model_config_against_plan(
line 911: def mlx_memory_breakdown_for_contract(
    ...
    """Recompute the portable MLX memory contract from normalized plan facts."""

planning.py:205 only wraps it: `calculated = mlx_memory_breakdown_for_contract(...)` inside `_mlx_memory_
- **Fix**: Extend the Owns cell to name the two responsibilities that actually live there: the portable MLX unified-memory estimator (`aptus-memory-mlx-v2`) and quantized-storage arithmetic, and the model-architecture / MoE-topology / quantization-layout contract validated against the loaded config both host-side and inside every bundle. The "Must not silently decide" cell should stay as-is.

#### `docs/guides/model-dataset-hardware.md:112` — DRIFTED (root cause J)

The fact guide scopes the 8 GiB Apple reserve floor to a local API scan, but both the CLI and the API apply it to manually declared facts whenever the backend is mps or an MLX runtime is selected.

- **Doc says** (`docs/guides/model-dataset-hardware.md:112`): A local API scan raises an explicitly selected Apple runtime reserve to at least 8 GiB.
- **Code does** (`src/aptus/api.py:417`): uses_unified_memory = (
        request.hardware.backend == Backend.MPS
        or request.target.training_runtime in {"mlx-lm", "pytorch-mps"}
        or (request.hardware.discovery == "local-scan" and sys.platform == "darwin")
    )
    if uses_unified_memory:
        reserve_gib = max(reserve_gib, 8.0)   # api.py:423; cli.py:547-548 does the same on --backend mps
- **Harm**: This is the section that tells a user which hardware facts to declare, including "per-device reserve". A user who declares `--reserve-gib 2` with `--backend mps` — or manual (non-scan) MPS facts through the API — silently gets 8.0, so their usable-memory arithmetic is 6 GiB off from what they entered. They then misdiagnose an infeasible or unexpectedly small candidate as an estimator bug instead of a floored reserve, and the doc's "local API scan" scoping actively tells them the floor does not apply to their manual path.
- **Fix**: Rewrite as: "Aptus raises the per-device reserve to at least 8 GiB whenever the declared backend is `mps`, the selected training runtime is `mlx-lm` or `pytorch-mps`, or a local scan runs on Darwin — for manually entered facts as well as scans, in both the CLI (`--reserve-gib`) and the API. MLX runtime admission re-applies the same 8 GiB floor."

### Low severity (3)

#### `docs/architecture/execution-orchestrator.md:26` — DRIFTED (root cause F)

The doc says train jobs launch either `run.py` or a CUDA Accelerate command, but single-device CUDA train jobs launch `train.py` directly with the plain interpreter and no Accelerate.

- **Doc says** (`docs/architecture/execution-orchestrator.md:26`): skips are rejected. Train launches the runtime-selected `run.py` or CUDA
- **Code does** (`src/aptus/execution.py:2563`): training_entrypoint = "run.py" if runtime_id == "mlx-lm" else "train.py"
- **Fix**: Describe all three launch shapes: MLX train launches `run.py`; single-distribution CUDA train launches `train.py` directly under the resolved interpreter (`return [interpreter, *train_arguments]`); distributed CUDA train launches `train.py` through the interpreter-bound `accelerate.commands.accelerate_cli launch --config_file config/accelerate.yaml`. Note that for managed CUDA jobs `JobService` — not `run.py` — is the parent that owns completion verification.

#### `docs/reference/api.md:628` — DRIFTED (root cause I)

api.md's per-status "Emitted errors" table lists only two 403 codes; the service also returns `403 desktop_execution_disabled` from `POST /api/v1/jobs` and from `POST /api/v1/validate` with `run: true`, and api.md never mentions the execution-disabled mode anywhere.

- **Doc says** (`docs/reference/api.md:628`): | `403` | `path_forbidden`, `desktop_session_required` |
- **Code does** (`src/aptus/api.py:1592`): @app.post("/api/v1/jobs", response_model=JobResponse)
    def create_job(request: JobRequest) -> dict[str, Any]:
        if not execution_enabled:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "desktop_execution_disabled",

(the same 403 is raised at api.py:1504-1506 inside `POST /api/v1/validate` before the `runtime_validat
- **Fix**: Add `desktop_execution_disabled` to the 403 row at api.md:628, and add one sentence to the `POST /api/v1/jobs` section (near the `409 runtime_unavailable` sentence at line 505) and to `POST /api/v1/validate` (near line 480) noting that a service started with local execution disabled rejects every job submission and every `run: true` runtime level with `403 desktop_execution_disabled` before the 409 paths are reached. Filed as low because error-codes.md:49 already defines the code and its meaning — the defect is confined to api.md's status-to-code enumeration and endpoint prose.

#### `docs/reference/method-registry.md:53` — MISSING (root cause D)

The descriptor schema table claims to list every field of the published aptus.method-descriptor.v1 payload and transcribes the dataclass field-for-field, but drops runtime_bindings, the field that actually determines MLX-LM executability.

- **Doc says** (`docs/reference/method-registry.md:53`): | `aliases` | Reserved descriptive aliases |
- **Code does** (`src/aptus/methods/contracts.py:68`): runtime_bindings: tuple[RuntimeBinding, ...] = ()
    schema_version: str = "aptus.method-descriptor.v1"
- **Fix**: Insert a `runtime_bindings` row between `aliases` and `schema_version` (matching dataclass order): "Executable (training runtime, compute backend) bindings, each carrying compiler_id, estimator_id, export_kind, evidence_requirement, and supported_distributions under `aptus.runtime-binding.v1`." api.py:771 serializes descriptors with to_primitive, which emits every dataclass field, so this field is published in capabilities.method_catalog; the OpenAPI schema types method_catalog as a free-form object, making this table the only field-level contract clients have.

---

## Closing judgement

**1. Unexamined changed docs.** The only files in the changed-`*.md` set never opened in any round are `dev/active/aptus-product-review/*` and `dev/active/moe-compatibility/*` (internal working notes, not a published surface), plus the root-level `README.md`, `ROADMAP.md`, `SECURITY.md`, `CHANGELOG.md`, `examples/README.md`, `desktop/macos/README.md` if rounds 1-2 skipped them. I just grepped all six: every one is MLX-aware (README:7 leads with "Apple Silicon or CUDA"; SECURITY:108 documents MLX admission; ROADMAP:59-71 owns the MoE and CUDA-vs-Apple boundary). **None that matter.** Every changed doc in `docs/` was read in full by some round; `docs/guides/design-an-evaluation.md` and `docs/guides/index.md` were never fully read but are also not in the changed set, so they cannot carry timeline drift.

**2. Final root-cause count: 12.** Round 3 added one genuinely NEW root cause — **L: MLX-LM's refusal of whole-text/`{"text": ...}` rows is undocumented anywhere in `docs/`** (finding 3, filed rc=none; grep-confirmed zero owner repo-wide). The other six round-3 findings are new *instances* of A (four instances), B, and J. Total confirmed: 34 findings / 12 root causes.

**3. Fix order.**
1. **A** — `choose-a-method.md:19` and `compare-plans.md:23`. Four confirmed instances, the file `docs/index.md:16` routes method selection to, zero MLX mentions in it, and it tells the only evidence-backed platform that it is unsupported. Highest harm, smallest edit.
2. **B** — add an MLX unified-memory admission entry to `troubleshooting.md:55` and correct `compile-validate-run.md`, `model-dataset-hardware.md`, `choose-your-path.md`. The one hard failure that actually blocked the recorded 2026-07-28 run, currently mis-triaged toward network/credentials.
3. **L** — disclose the whole-text refusal at `prepare-a-dataset.md:37`/:18. Wastes an entire human-review pass, no owner.
4. **J** — rescope the 8 GiB reserve floor at `model-dataset-hardware.md:112` to all mps/MLX paths, not just scans.
5. **C**, then **D/E/F/G/H/I/K** — reference/contributor surfaces; harm is confusion, not wrong hardware decisions.

**4. Verdict: CLOSED.** No further check required. Coverage of the changed-doc set is complete, the three rounds converged (round 3 rejected 17 of 24 candidates and added one root cause, not a new class of defect), and the residual unread files are provably outside the drift window.
