# Aptus Examples

> **Status:** Active | **Authority:** Example guide | **Applies to:** Aptus 0.2 | **Audience:** New users and test authors | **Last reviewed:** 2026-07-22 | **Review by:** 2026-10-22 or when example data changes

`support-sft.jsonl` is a small prompt-completion dataset for exercising Aptus
planning, compilation, and static validation. It is synthetic example data. It
is not a quality benchmark and should not be used to evaluate a model.

Profile it with:

```bash
aptus profile \
  --dataset ./examples/support-sft.jsonl \
  --sequence-length 128 \
  --output ./aptus-work/dataset-profile.json
```

The example supports contract and static product checks without downloading a
model. Runtime validation and training still require a real immutable model
revision, training rights, compatible dependencies, and supported CUDA
hardware.

`reviewed-sft.jsonl` demonstrates the governed-feedback shape described in the
[reviewed corpus contract](../docs/reference/reviewed-corpus-contract.md). Its
rows are synthetic. Two rows share each `split_group`, so the generated trainer
must keep the related source material on one side of the train/evaluation
boundary.

## Related documentation

- [First planning-only run](../docs/getting-started/first-plan.md)
- [Dataset schemas](../docs/reference/dataset-schemas.md)
- [Prepare a dataset](../docs/guides/prepare-a-dataset.md)
