# GridOps Case Study Visual Assets

Generated with the Gemini API for the Capabl Machines GridOps case study.

Source script:

```bash
set -a; . ./.env; set +a
.venv/bin/python scripts/generate_case_study_images_gemini.py
```

Official Gemini image-generation docs used:

- https://ai.google.dev/gemini-api/docs/image-generation
- https://ai.google.dev/gemini-api/docs/imagen

## Assets

Use the `.webp` versions in web pages. Keep the `.png` files as source-quality originals.

| Web asset | Source | Purpose |
|---|---|
| `gridops_hero_microgrid.webp` | `gridops_hero_microgrid.png` | Case-study hero: Indian community microgrid, solar, battery, grid context. |
| `gridops_control_room.webp` | `gridops_control_room.png` | Operational expertise visual: engineers monitoring microgrid dispatch. |
| `gridops_environment_loop.webp` | `gridops_environment_loop.png` | Environment/model/action loop visual for architecture sections. |
| `gridops_impact_split.webp` | `gridops_impact_split.png` | Impact visual contrasting do-nothing vs trained-model operation. |
| `india_solar_society_hero.webp` | `india_solar_society_hero.png` | India-context hero: apartment society, rooftop solar, battery, EV charging, and local dispatch. |
| `india_microgrid_operator_layer.webp` | `india_microgrid_operator_layer.png` | RWA/society operator supported by an AI intelligence layer. |
| `india_microgrid_value_flows.webp` | `india_microgrid_value_flows.png` | Visual metaphor for savings, reliability, and earning potential from local flexibility. |

These are editorial visuals for storytelling. For exact metrics and evidence, use the committed eval plots in `evals/plots/`.
