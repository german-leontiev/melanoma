# Model verification

The public model definition was checked on 21 July 2026 against the private
project checkpoint before the original repository was retired.

| Check | Result |
| --- | --- |
| Restricted checkpoint inspection | 354 state-dict tensors; no arbitrary checkpoint code executed |
| Strict state-dict load | Passed with no missing or unexpected keys |
| Parameter count | 26,390,707 |
| Input/output smoke test | `(1, 3, 600, 600)` → `(1, 1)` |
| Production export parity | 100/100 identical classes between PyTorch and ONNX Runtime |
| Maximum recorded probability delta | `2.413988e-06` |

The checkpoint, source images and ONNX weights are intentionally not included.
They remain covered by the private encrypted archive and the operational
Derma AI project. The parity result validates the production export against
that checkpoint; it does not validate diagnostic quality.
