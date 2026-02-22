# Example: Medical AI Pipeline (Myocardial Strain Quantification)

This is a real-world example of the agent-based parallel development strategy applied to a cardiac MRI analysis system.

## Project Summary

An end-to-end deep learning pipeline that takes cardiac MRI as input and outputs myocardial strain values for risk stratification after heart attack.

## Results

- **6 agents** ran in parallel
- **6,446 lines** of production code generated
- **32 files** created/modified
- **224 tests** passing after integration
- **1 bug found** at module boundary (CarMEN decoder skip connection), fixed in minutes

## Phase Decomposition

```
Phase 1: Data Pipeline ──────┐
                              ├── Phase 4: Strain Computation ── Phase 5: Risk ── Phase 6: API
Phase 2: Segmentation ───────┤
                              │
Phase 3: Motion Estimation ──┘
```

Phases 1, 2, and 3 ran fully in parallel. Phases 4-6 were also launched in parallel since interfaces were defined upfront.

## Agent Instruction Files

Each file in this directory was the complete instruction set for one parallel agent:

| File | Phase | Lines of Code Produced |
|------|-------|----------------------|
| `phase1_data_pipeline.md` | Data loading, preprocessing, augmentation | ~1,267 |
| `phase2_segmentation.md` | U-Net model, training loop, metrics | ~1,240 |
| `phase3_motion_estimation.md` | Registration network, displacement fields | ~932 |
| `phase4_strain_computation.md` | Biomechanical strain tensors, AHA model | ~976 |
| `phase5_risk_stratification.md` | Risk models, clinical reports | ~1,222 |
| `phase6_api_deployment.md` | FastAPI, Docker, orchestration | ~809 |

## Lessons Learned

1. **The bug was at a boundary** — CarMEN's decoder tried to add tensors with mismatched channels (Phase 3 internal issue, but caught because Phase 3 output needed to feed Phase 4)
2. **Synthetic data unlocked testing** — real data was behind a firewall, so a generator script let us validate the full pipeline
3. **Interface contracts prevented 95% of integration issues** — agents that followed the contracts produced compatible code
4. **Agent instruction files became the best documentation** — more precise than any design doc we would have written manually
