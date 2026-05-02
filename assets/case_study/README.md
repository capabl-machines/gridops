# GridOps Case Study Visual Assets

Generated with the Gemini API for the Capabl Machines GridOps case study.

Source script:

```bash
set -a; . ./.env; set +a
python scripts/generate_case_study_images_gemini.py
```

Official Gemini image-generation docs used:

- https://ai.google.dev/gemini-api/docs/image-generation
- https://ai.google.dev/gemini-api/docs/imagen

## Assets

| File | Purpose |
|---|---|
| `gridops_hero_microgrid.png` | Case-study hero: Indian community microgrid, solar, battery, grid context. |
| `gridops_control_room.png` | Operational expertise visual: engineers monitoring microgrid dispatch. |
| `gridops_environment_loop.png` | Environment/model/action loop visual for architecture sections. |
| `gridops_impact_split.png` | Impact visual contrasting do-nothing vs trained-model operation. |

These are editorial visuals for storytelling. For exact metrics and evidence, use the committed eval plots in `evals/plots/`.
