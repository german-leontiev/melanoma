# Legacy experiment record and evidence limits

The private 2025 experiment compared a baseline ResNeXt50 classifier with a
ResNeXt50 model augmented by CBAM in stages three and four. The archived
notebooks recorded the following values on a 100-image binary holdout:

| Model | Accuracy | F1 | F2 |
| --- | ---: | ---: | ---: |
| ResNeXt50 baseline | 0.8500 | 0.6667 | 0.6250 |
| ResNeXt50 + CBAM | 0.8700 | 0.7451 | 0.7540 |

The notebook also recorded a paired bootstrap non-improvement share of 0.124
for F1 and 0.015 for F2 at threshold 0.46. The implementation in
`melanoma_research.metrics` retains paired resampling but names the quantity
explicitly instead of presenting it as a universally valid clinical p-value.

These numbers are historical evidence, not a current benchmark:

- the repository does not contain the 100-image holdout or its immutable
  manifest;
- patient/lesion identifiers were not preserved with the archived folder-based
  training corpus, so leakage cannot be ruled out retrospectively;
- the threshold was evaluated on the same recorded holdout;
- the sample is too small for claims about clinical safety or population-level
  performance;
- comparison with physician metrics from a separate study is not a paired,
  like-for-like clinical evaluation.

For those reasons the public repository does not claim that the model
outperforms a physician. A new experiment should freeze a patient- or
lesion-grouped split, define the threshold before final evaluation, report
sensitivity/specificity and uncertainty, and use an external cohort.
