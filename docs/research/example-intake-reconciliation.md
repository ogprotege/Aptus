# EXAMPLE forensic review and salvage ledger

> **Status:** Archived | **Authority:** Historical provenance ledger | **Snapshot:** 2026-07-22 | **Audience:** Maintainers and researchers | **Review:** Only when the recorded intake or disposition is challenged

Status: all 75 local intake files reviewed on 2026-07-22. This ledger covers 58
non-PDF files and 17 PDFs. The raw `EXAMPLE/` folder is ignored and remains
local. None of its code, credentials, databases, or unpublished material is
part of the repository change.

## What the work was trying to accomplish

The material shows four connected lines of work:

1. Fine-tune Llama 3.3 70B on a Lambda multi-H100 host with LoRA, BF16,
   gradient checkpointing, and FSDP.
2. Reduce the trainable set further through BitFit, DoRA, LoReFT, AFLoRA, and
   combinations of those ideas.
3. Capture specialist conversations, corrections, and ratings through a
   Chainlit application so a reviewed corpus could improve later runs.
4. Add telemetry, validation, checkpointing, testing, and deployment around
   the training process.

That product direction belongs in Aptus. The historical implementations do
not. They contain syntax failures, incompatible runtime combinations, custom
adapter math that does not match the papers, weak data identity, incomplete
evaluation, insecure feedback storage, and success paths that can conceal
failure.

## Central BitFit finding

BitFit was a reasonable method to investigate. The selected model family was
the problem. The scripts froze the model, selected parameter names containing
`bias`, and continued without proving that the result was non-empty. Default
Llama attention and MLP projections are bias-free, and RMSNorm has no bias. A
Llama 3.3 70B BitFit run could therefore train few or zero parameters.

Aptus now performs a fail-closed census immediately after method preparation.
It requires unique names, positive tensor and parameter counts, finite values,
and a stable digest over sorted trainable names, shapes, and dtypes. Full tuning
rejects frozen model tensors. The current LoRA-based paths reject trainable
tensors outside the compiled LoRA scope. Measured preflight, pilot, and full-run
metrics carry the census without exposing names or values, and the two pilot
phases must agree. BitFit remains experimental until an exact architecture
exposes a meaningful existing bias set and a bias-delta save and reload path
passes.

## Salvage implemented now

| Historical intent | Aptus implementation |
|---|---|
| Count and log the exact trainable set | Generated model-data, synthetic, pilot, and training paths require a positive finite method-scope census; measured preflight, pilot, and full-run metrics record its descriptor digest |
| Prevent same-source chunks from inflating evaluation | Full training keeps each explicit top-level or `metadata.split_group` value on one side, detects canonical-data mutation, requires distributed digest agreement, and records target, realized, row, group, unit, dataset, and assignment evidence |
| Separate supported methods from interesting papers | `src/aptus/methods/` and the workbench readiness board expose four selectable `gated-executable`, four nonselectable `experimental`, and three nonselectable `research-only` descriptors |
| Preserve DoRA, BitFit, AdaLoRA, LoReFT, AFLoRA, BiLoRA, and ShareLoRA ideas honestly | Every method has a primary evidence identity, mechanism, blocker, and required proof without becoming selectable |
| Capture expert corrections for later training | The [reviewed corpus contract](../reference/reviewed-corpus-contract.md) defines immutable IDs, consent, provenance, redaction, human approval, grouping, and approved-only export |
| Use the actual M5 Pro for local experiments | The local probe records Apple Silicon and shared unified memory; the [Apple pilot matrix](../operations/apple-silicon-pilot.md) names exact proposed models and methods while the CUDA compiler remains fail-closed |
| Preserve run telemetry and no-clobber output | Existing Aptus run IDs, leases, logs, measured memory, checkpoint continuation, parent verification, and structural export checks already supersede the historical scripts |

## Method conclusions

| Method | Evidence conclusion | Aptus lifecycle |
|---|---|---|
| LoRA | Strong primary evidence and the historical r64, alpha128, all-projection choice is useful as experiment history, not a universal default | Gated executable |
| QLoRA | Appropriate local memory-saving recipe when the selected runtime and quantized model pass exact probes | Gated executable on CUDA; proposed MLX pilot only |
| DoRA | Strong primary evidence. Implement through maintained PEFT or MLX-LM, never the custom files in this intake | Experimental until compiler, estimate, save, reload, and pilot pass |
| BitFit | Valid bias-only method with strongest supplied evidence on encoder classification. Architecture-dependent and likely empty on stock Llama | Experimental, exact bias census required |
| AdaLoRA | Strong adaptive-budget evidence. Requires schedule and importance state in checkpoints | Experimental |
| LoReFT | Representation intervention, not a LoRA weight wrapper. Requires separate collator, runtime hook, artifact, and inference contract | Research only |
| AFLoRA | Dynamic freezing requires score, schedule, optimizer-membership, and restart state | Research only |
| BiLoRA | Bilevel method needs disjoint D1 and D2 training partitions, two optimization levels, and higher-order memory measurement | Research only |
| ShareLoRA | Promising shared-factor topology. Serialization and distributed ownership need verification | Experimental |
| AFLoRA plus LoReFT | A new composed strategy, not a free combination. The supplied hybrid is neither paper-faithful nor runnable | Rejected until both methods work independently and the composition has its own contract |

## Corpus and evaluation conclusion

The Chainlit application should not become an Aptus service. It stores raw
content and identifying data without the needed consent, retention, redaction,
review, access, and export controls. Ratings can attach to the wrong message,
and raw model output can flow into exports without a correction contract.

The useful design is a governed intake service with immutable interaction and
turn IDs, model and prompt provenance, source citations, reviewer and rubric
identity, doctrinal and factual labels where appropriate, PII adjudication,
approved corrections, deduplication, and split groups. Raw thumbs-up feedback
must never trigger training automatically.

## Complete non-PDF ledger

`Salvage` means the named requirement was carried into Aptus. `Archive` means
the file is useful only as historical intent. `Discard` means no implementation
or factual claim should be imported. The local source remains untouched unless
the user later chooses to clean the intake copy. Hashes record the bytes seen at
review time. They are not assertions about the current ignored `EXAMPLE/` tree,
which the user may continue to edit. Finder `.DS_Store` files are volatile junk,
so their snapshot hashes are provenance only and must not be used as live
identity checks.

| Path | SHA-256 | Disposition | Finding |
|---|---|---|---|
| `EXAMPLE/.DS_Store` | reviewed snapshot `4feb8a2682c6e24987053db816a1c60ed83f0d9bacde080d79ee69848136844a`; live hash intentionally untracked | Discard | Volatile Finder metadata only |
| `EXAMPLE/5-10-25/README.md` | `2004d4c7fc8d810804646ad81268f30c6774c43398a5b1a98643c3ef04cba528` | Archive | Describes a nonexistent production-ready tree; LR search, calibration, bias monitoring, and dashboards survive only as future requirements |
| `EXAMPLE/5-26-25/prompt.txt` | `7d3ba5f9c408aff140b14edbc4689773430ccea4aff3481a02fa1516addf196a` | Salvage | Tester-gated, chronological, human-corrected corpus intent maps to the reviewed corpus contract |
| `EXAMPLE/Chainlit_05-03-25/README.md` | `b15d71b4446bfe19522e5ab6e45dbab2b436efd5bb9726e8e362d17f2e6ef86e` | Discard | Security and completeness claims exceed the application |
| `EXAMPLE/Chainlit_05-03-25/analyze_data.py` | `1663d2b298e521d62c251c1b8c3874e379972e9f7ebe550204e082b0f13bbe7a` | Salvage concept | Latency and rating summaries are useful; replace them with dataset quality, reviewer agreement, and task metrics |
| `EXAMPLE/Chainlit_05-03-25/app.py` | `7d49def2232d7c3a7fac49792349a75b9247e44163e459a750711daca2cfecd8` | Discard code | Raw content and IP storage, weak passwords, credential logging, duplicate hooks, feedback misassociation, and no governance boundary |
| `EXAMPLE/Chainlit_05-03-25/backup_db.py` | `913cbc99dcd9f23fc20da2ecf0da1b61e9a461f48c4ab8740b2960cb060abf39` | Salvage concept | Backups need the SQLite backup API, integrity checks, encryption, retention, and access policy |
| `EXAMPLE/Chainlit_05-03-25/chainlit.md` | `eb64d3585ba662bb6efb45c50746117e37f9305d03f88ec22483dda6357e249a` | Discard | Privacy statement does not disclose raw content and IP collection |
| `EXAMPLE/Chainlit_05-03-25/chainlit_info_v1.md` | `11e2c8ac90bbc1c0d11263fdc025a49f142f87d0f625e2ba3d6b29e84909dcd7` | Discard | Generated inventory wrapper adds no reliable contract |
| `EXAMPLE/Chainlit_05-03-25/chainlit_stepbystep.md` | `bb52d889d5a1f221e36c9a86ea567b5dc09e13dcfe9e06eb4c3f0e5f1b7d0612` | Discard duplicate | Byte-identical to `detailed-installation-guide.md` |
| `EXAMPLE/Chainlit_05-03-25/chainlit_with_together.py` | `49bf7e7d9805972c3e0c00b520bebe678e60dac39c92f496993ba2b5412b9462` | Discard code | Simplified variant retains the core data and security defects |
| `EXAMPLE/Chainlit_05-03-25/comprehensive-guide.md` | `0f090806c8c3af21fc412d3af7c0469e6345cc042a4415175bea8aa179634c1b` | Discard duplicate | Byte-identical to `comprehensive_guide.md`; governance intent is preserved elsewhere |
| `EXAMPLE/Chainlit_05-03-25/comprehensive_guide.md` | `0f090806c8c3af21fc412d3af7c0469e6345cc042a4415175bea8aa179634c1b` | Discard duplicate | Same content as the hyphenated file |
| `EXAMPLE/Chainlit_05-03-25/detailed-installation-guide.md` | `bb52d889d5a1f221e36c9a86ea567b5dc09e13dcfe9e06eb4c3f0e5f1b7d0612` | Discard duplicate | Byte-identical to `chainlit_stepbystep.md` |
| `EXAMPLE/Chainlit_05-03-25/env.example.txt` | `43fe9cc3044f767a2fa02ca2762e5188fcfa888881931144e60ca424b9cc81a3` | Salvage concept | Secrets belong in environment or credential stores, never plan or script fields |
| `EXAMPLE/Chainlit_05-03-25/export_data.py` | `0a0106f76c247e0b95a3433be105fb0f2ab09f1a3748f891fb73a8ce31df0f52` | Discard code | Raw identifying export is unsafe; approved SFT and preference exporters need hashes and lineage |
| `EXAMPLE/Chainlit_05-03-25/gitignore.txt` | `c05e37c183b58c14cebb490ec1080ef9f333479aa413f959fcb16247258ebd99` | Salvage selectively | Credential, database, and cache exclusions are useful; the broad JSON exclusion is not |
| `EXAMPLE/Chainlit_05-03-25/huggingface_spaces_README.md` | `7df5e617b55438bdac150b31e318f39fb0e3e0cae8e739e02b926291bb2c23ac` | Discard | Stale deployment recipe with unsupported security claims |
| `EXAMPLE/Chainlit_05-03-25/requirements.txt` | `ec593ce9d476ed92b50fb1a5f1c3ebc90cfa3a9942609d6322b6bb6aa25dfd5a` | Discard | Stale unlocked application dependency set |
| `EXAMPLE/Chainlit_05-03-25/simple-deployment-plan.md` | `b99d471d0b356ce47d91dfa0330f1db3b888cca028196ce73f435f6a1fe4fae3` | Discard | No auth, storage, retention, or worker isolation contract |
| `EXAMPLE/Chainlit_05-03-25/test_together_api.py` | `8d00606e23dcd1f41f8066cfa5e7faabe288a069e38b344bcf8b872caa296369` | Salvage pattern | Provider smoke tests should cover streaming, nonstreaming, latency, assertions, and token accounting |
| `EXAMPLE/Chainlit_05-03-25/text 2.txt` | `0d141957830ba62c5b761ee00a5e3c44815b12d3ed2d4c676109aec67a144d6d` | Discard | Empty or whitespace-only junk |
| `EXAMPLE/Chainlit_05-03-25/text 3.txt` | `691f0783706990a5a89644e2d41550ceaea21b765424b89fd2770764c1a0a6a1` | Discard | Empty or whitespace-only junk |
| `EXAMPLE/Chainlit_05-03-25/text.txt` | `6e31d965edfefe25d29131fbe444e1bd28e75004861b5f581b5b7d42bdab76fd` | Discard | Empty or whitespace-only junk |
| `EXAMPLE/FT-New_4-20-25/.DS_Store` | `e4902606c6f2e1f66229e3f110ce031e0c9e6a062109ac0dab78ffc6f38abafd` | Discard | Finder metadata only |
| `EXAMPLE/FT-New_4-20-25/Finals/ft-phase1-streamlined-v2-70b.py` | `de258fa07cb2683cbe5d98483eb24e97539ec46c69dd24b212da4d26bb1d036c` | Discard code | Invalid or nonserializable FSDP configuration and no trustworthy evaluation or resume proof |
| `EXAMPLE/FT-New_4-20-25/Finals/gemini-ft-phase1-pro-70b.py` | `8c3175e575ca3f1014b88005fbae5e8c71fd7d05fe24296960c3a92baea67bf6` | Salvage intent | Run summary, effective batch, and telemetry survive; runtime configuration does not |
| `EXAMPLE/FT-New_4-20-25/Finals/mistral-ft-phase1-v2-70b.py` | `16da126f273ec1b89392fbe9d58d96ea65190a02d2b293bc22fa5131a4259a2d` | Discard code | Broken argument reconstruction and unsafe credential handling |
| `EXAMPLE/FT-New_4-20-25/Finals/o1mini-ft-phase1-v2-70b.py` | `ec0133252b41b120b1c9c9221d4b930a020e5ee99a5b0c0cf0e5b17785301caa` | Archive | Strongest resume and metric intent, but checkpoint discovery and full-logits evaluation are unsafe at 70B |
| `EXAMPLE/FT-New_4-20-25/Finals/qwen-ft-phase1-mod-70b.py` | `117d812fb1bf277b8dd214f7a0653573a90a13f7d02a97a6f33faff928f653c8` | Discard code | Immediate import failure plus broken argument reconstruction |
| `EXAMPLE/FT-New_4-20-25/V1/ft-phase1-streamlined-70b.py` | `fb21b08d9b02634e525a64736371346b750866be8acee633805be5fa1a8b0abc` | Salvage intent | Timestamp, no-clobber, validation discovery, memory, and elapsed-time ideas survive |
| `EXAMPLE/FT-New_4-20-25/V1/gemini-ft-phase1-v2-70b.py` | `da0df5c503fec54b8ceca988438a9a381dbbf3333b675a0f93c7697b0a484032` | Archive | Best early structure, but unverified imports and dataset behavior prevent reuse |
| `EXAMPLE/FT-New_4-20-25/V1/mistral-ft-phase1-70b.py` | `36e2443fcee78ab7b7c07d5ad32a24e1f73096ed270cc0f6a096e1102cfbc568` | Discard code | Immediate missing-import failure plus broken arguments |
| `EXAMPLE/FT-New_4-20-25/V1/o4mini-ft-phase1-70b.py` | `3eaadb63923e17ec5cd8cfe6d42c459aadab4608b0d07a35e5431a234509ffd3` | Salvage intent | Explicit CLI facts and fail-closed behavior survive; full-logits perplexity is a 70B memory hazard |
| `EXAMPLE/FT-New_4-20-25/V1/qwen-ft-phase1-70b.py` | `85d560deece8a5ea2009b5c0469b44cf6863d42c6756f8c8543884d902f4844e` | Discard code | Minimal prototype can swallow failure and does not bind artifacts |
| `EXAMPLE/FT-_05-17-25/AFLoRA-LoReFT Implementation Instruction.md` | `4b074e6c7b8267a2c1a49c17622428ecc91132ac180c772fc612b8d3225fab82` | Discard duplicate | Byte-identical to the copy under `New_FT_05-16-25` and describes an invalid hybrid |
| `EXAMPLE/FT-_05-17-25/bitfit_instructions_v1.md` | `e96b200ce2b0c232f71b41d63a02db875c592ad79c8a6fc03131bb47d09e9c00` | Archive | Historical BitFit intent does not match stock Llama bias structure |
| `EXAMPLE/FT-_05-17-25/bitfit_instructions_v2.md` | `f36352f103b8eead0ff14b70277570551dfdc2a398e33669733be936145762b8` | Discard recipe | Unverified stack claims and invalid RoPE procedure |
| `EXAMPLE/FT-_05-17-25/bitfit_v1.py` | `4678d55d1635409292f8ee2207988dd9c679561dffe67202dc91d1dc8fa5663d` | Discard code | No positive trainable assertion, device-map and FSDP conflict, invalid callbacks |
| `EXAMPLE/FT-_05-17-25/bitfit_v2.py` | `0dbad4ed8c3cd3fe31466514a83bf7fea2deae7f15c4bb0bbea5540e28fbfd64` | Discard code | Syntax-invalid duplicate keyword plus the same architecture defects |
| `EXAMPLE/FT-_05-17-25/bitfit_v3.py` | `8469539436b28204b8cf029a28ec0cb5fec714eb0332f9d2d7a324870e35ece9` | Salvage census only | Trainable-name reporting is useful; the RoPE rewrite and runtime remain invalid |
| `EXAMPLE/New_FT_05-16-25/AFLoRA-LoReFT Implementation Instruction.md` | `4b074e6c7b8267a2c1a49c17622428ecc91132ac180c772fc612b8d3225fab82` | Discard | Invalid hybrid recipe and exact duplicate of the earlier copy |
| `EXAMPLE/New_FT_05-16-25/aflora_loreft_v1.py` | `94c1450bc203490b268e27e50f30326f479a10a08404944f274373df08fe7bcc` | Discard code | Dimension failures, unused position logic, and unreliable gradient-based freezing |
| `EXAMPLE/New_FT_05-16-25/dora_draft.py` | `9b51c087275488bf1b7601919459e7543060df46aed4c17ce2ad0e9c7f2d24ea` | Discard | Syntax-invalid Markdown and Python mixture |
| `EXAMPLE/New_FT_05-16-25/dora_instructiosn_another.txt` | `36923dfc3feccb75161b6f84a1da93ee6345112fc05591989fd3356be75735aa` | Discard | Operational recipe depends on the invalid custom backend |
| `EXAMPLE/New_FT_05-16-25/dora_setup_instructions.md` | `f824f64e99017f2bbd6cea79682bd1b15f3eb08519485de2d43225c957d7f14c` | Salvage requirements | Preflight, checkpoint, and telemetry requirements are already stronger in Aptus |
| `EXAMPLE/New_FT_05-16-25/doraft_v1.py` | `0837babddfed8e0fdb11abfd8228ddc8e37cbd64647776ecde703976b6b0bcd5` | Discard code | Custom math is dimensionally incorrect and not paper-faithful DoRA |
| `EXAMPLE/New_FT_05-16-25/gemini_loreft_instructions_v1.md` | `90e9811c0e2f646c7739487a1e7bfcc83002491a789ed32126449165a062e3f2` | Archive | Points toward pyreft but lacks a verified intervention and export contract |
| `EXAMPLE/New_FT_05-16-25/gemini_loreft_v1.py` | `79ee5aeed92365dc61a1c62e9acc3d172cf4fa2bfdc716eb2848162dcd031610` | Archive | Only implementation aimed at official pyreft; API, collator, location, and export behavior remain unproved |
| `EXAMPLE/New_FT_05-16-25/grok_loreft_v1.py` | `a37eed858e4254039462be4647b0ba28fc937f5a6ef4ab6bbc5cc6e24b9a3874` | Discard code | Wrong projection core plus conflicting distributed configurations and blind retries |
| `EXAMPLE/New_FT_05-16-25/grok_loreft_v1_instructions.md` | `b3455d56eef7aecfe1ac65edade46c89ddc80a47a0d060499b51a522071b0830` | Discard | Recipe depends on invalid runtime |
| `EXAMPLE/New_FT_05-16-25/instructions-loreft_v3.txt` | `4bd7f2e440011ae3fb19bf3965026d1cb8c4fcdb48e723802f345d44fd0b48e6` | Discard | Stale system and package recipe |
| `EXAMPLE/New_FT_05-16-25/instructions_loreft_v5.md` | `4187c93a66b57315ba599d8b5fe95005bbdf063dfe1ed54ed8e6984525c7ff7e` | Discard | Unsupported safety and completeness claims tied to the wrong implementation |
| `EXAMPLE/New_FT_05-16-25/intsructios_loreft_v4.txt` | `8f9636ee768fd38dd8e84089d4ad0cc5981bb82f2256e31632f870edec706199` | Discard | Stale recipe for invalid custom code |
| `EXAMPLE/New_FT_05-16-25/loreft_v3.py` | `77df0aee049ce1525be7a4bfbedc91c77eb2b56390763247f1f330b8e1861756` | Discard code | Custom q/v projection and FFT wrapper is not LoReFT |
| `EXAMPLE/New_FT_05-16-25/loreft_v4.py` | `02aa99151d7da6d173721fd9f794a76aedff1e69d74969e8388f5a52fa922894` | Discard code | Same incorrect mechanism despite added checks |
| `EXAMPLE/New_FT_05-16-25/loreft_v5.py` | `88daaf5b894669f5aa09a9aced5a63639bfcb80c49249bc1474f95cabc4f8a07` | Discard code | Large implementation with wrong core math and weaker artifact guarantees than Aptus |
| `EXAMPLE/finetune_lora_70b.py` | `aacd9e3edaf436d4fd18013633bda0b1bf3bc441ab0df0516a13b7603ab41b63` | Salvage history | Confirms the LoRA r64, alpha128, attention-plus-MLP targets, BF16, checkpointing, and FSDP experiment history; code is not reusable |

## Complete PDF ledger

Every PDF was extracted in full and its first page was rendered for visual
verification. Paper results describe their own models, tasks, ranks, software,
and hardware. They are not Aptus capacity constants.

| Path | SHA-256 | Identity | Disposition and Aptus use |
|---|---|---|---|
| `EXAMPLE/Fine-Tuning/1-s2.0-S2949719125000202-main.pdf` | `e83d8a5f1a464ce6165f6b8b1efb232396549e56f0132bf482f2ecceec3ba3f3` | Pratap et al., *The fine art of fine-tuning*, 2025 | Retain as a secondary taxonomy and citation map; do not use heterogeneous comparison tables as calibration |
| `EXAMPLE/Fine-Tuning/1-s2.0-main.pdf` | `e79c012f98381800c33bc58bc5a7316a330341e936946d7b9dccdb8c80840403` | Same article as prior row | Discard duplicate; extracted content differs only by an Elsevier logo line and the first-page render is malformed |
| `EXAMPLE/Fine-Tuning/14bfffd6ccb59d615a65d40605c7af5c2f7a.pdf` | `725abeed0fffaddbed010784046b2f7514c747a4e42108139c96f3b787cd35da` | Ersoy and Ersahin, *Benchmarking Llama 3 70B for Code Generation*, 2024 | Low-confidence archive; reported scores omit the run facts needed for planning or comparison |
| `EXAMPLE/Fine-Tuning/1907.10902v1.pdf` | `a1ec86ba21a3f2e7bb774c00f8b108dc955993f39feed72f5d8ee3910b6d8a51` | Akiba et al., *Optuna*, arXiv:1907.10902 | Roadmap a bounded search layer over feasible plans with persistent trial provenance and validation-only objectives |
| `EXAMPLE/Fine-Tuning/2106.09685v2.pdf` | `e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a` | Hu et al., *LoRA*, arXiv:2106.09685 | Canonical LoRA evidence; expose rank, alpha, dropout, targets, merge policy, and measured memory |
| `EXAMPLE/Fine-Tuning/2303.10512v2.pdf` | `7afe399a4d0c5cf37c956b0a8afb2294c2d2b289608162b4e49bddecc3b311ef` | Zhang et al., *AdaLoRA*, arXiv:2303.10512 | Experimental registry entry; compiler must preserve adaptive budget, importance, schedule, and checkpoint state |
| `EXAMPLE/Fine-Tuning/2304.01933v3.pdf` | `42e5adc38d4a762077a7060d3c6b6d84d03c1fca132e3ab2581ef3c1245eac7f` | Hu et al., *LLM-Adapters*, arXiv:2304.01933 | Supports measured target and placement search; reported rank and placement choices are not universal defaults |
| `EXAMPLE/Fine-Tuning/2312.12148v1.pdf` | `16d7161b3ad5422e536b0dd5f92920d75c29ac8ad12e845c09966fa763798398` | Xu et al., critical PEFT review, arXiv:2312.12148 | Best supplied BitFit evidence, but on RoBERTa classification; use exact architecture eligibility and trainable-byte accounting |
| `EXAMPLE/Fine-Tuning/2402.09353v6.pdf` | `f048986e4d0b1c5b0a3f6c3c709ab245a5bcb993aa92704d6170304b413f08a4` | Liu et al., *DoRA*, arXiv:2402.09353, ICML 2024 | Strong method evidence; implement with maintained runtime, explicit magnitude state, merge validation, and measured pilot |
| `EXAMPLE/Fine-Tuning/2402.15061v2.pdf` | `526d7907fe8c28fe5c1be877231e7d72b69ee2e49c93a2f32215d580ee9be04d` | Zheng et al., *DragFT*, arXiv:2402.15061 | Treat as a translation corpus recipe with dictionary, retrieval, lineage, and domain metrics, not an adapter family |
| `EXAMPLE/Fine-Tuning/2403.13037v1.pdf` | `a23f69291209d041910b1bd673776ae0bd09110f4b1273df84305614725ae526` | Qiang et al., *BiLoRA*, arXiv:2403.13037 | Research-only bilevel compiler; needs D1 and D2 partitions, two optimizers, hypergradient memory, resume, and untouched test data |
| `EXAMPLE/Fine-Tuning/2403.13269v3.pdf` | `8e5e2a6db49a0be0ecd8309f15f45a35a02860f827b281ee7b2e6cbc253b6b07` | Liu et al., *AFLoRA*, arXiv:2403.13269 | Research-only dynamic-freezing compiler; preserve score, schedule, frozen-set history, optimizer state, and restart equivalence |
| `EXAMPLE/Fine-Tuning/2404.03592v3.pdf` | `0a05f0a333aaa5a660fcd04ebe773904a880168480990dd5e42ab5e0ec56e32b` | Wu et al., *ReFT and LoReFT*, arXiv:2404.03592 | Separate representation-intervention runtime and artifact; not mergeable LoRA weights |
| `EXAMPLE/Fine-Tuning/2406.10785v1.pdf` | `4e815a0f236a8040cbfc9b6b9d728104ff247bbdfa65a5e2cea3b2bc6443e143` | Song et al., *ShareLoRA*, arXiv:2406.10785 | Experimental shared-factor topology; require shape grouping, unique parameter accounting, serialization, and distributed synchronization |
| `EXAMPLE/Fine-Tuning/2408.13296v3.pdf` | `d5cb9007312a04536661b28b9e84bee495e963fb6032e5a68c2a1c3d2ebfd8ae` | Parthasarathy et al., exhaustive fine-tuning review v1.1 | Retain as a lifecycle checklist; reject its universal defaults and incorrect claim that QLoRA quantizes adapter weights |
| `EXAMPLE/Fine-Tuning/2408.13296v3v2.pdf` | `8e08c69874e74f07ae3e7676cb23e9135bdd86c126e8cc372f9a341b07d6bb29` | Byte-identical extracted text to prior row | Discard duplicate |
| `EXAMPLE/Fine-Tuning/Definitive Guide to Testing LLM Applications.pdf` | `3d23e9b5d0ae20070291c9e702863cad58750cbd2621f3ae9cace00f4c7dd497` | LangChain, *The Definitive Guide to Testing LLM Applications* | Salvage vendor-neutral test suites, evaluator versioning, human review, repeated judges, CI subsets, and production-failure regression intake; do not require LangSmith |

## What does not enter Aptus

- no custom DoRA, LoReFT, AFLoRA, or hybrid module from this folder;
- no historical dependency or deployment recipe;
- no raw Chainlit database, authentication, backup, or export code;
- no paper headline metric as a planner score;
- no hard-coded rank, learning rate, layer list, memory multiplier, or RoPE
  setting as a universal default; and
- no automatic training from captured model responses or ratings.

The old credential strings in historical scripts are intentionally omitted
from this ledger. The user identified them as obsolete. Keeping `EXAMPLE/`
ignored prevents the intake copy from entering a commit.

## Related documentation

- [Research index](index.md)
- [Reviewed corpus contract](../reference/reviewed-corpus-contract.md)
- [Apple Silicon experiment matrix](../operations/apple-silicon-pilot.md)
- [Retained Reference packet](../../Reference/README.md)
