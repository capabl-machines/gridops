# GridOps Case Study Visual Assets

Generated with the Gemini API for the Capabl Machines GridOps case study.
The `capabl_*` assets were generated directly with Codex image generation for
the India-focused case-study page; the earlier `gridops_*` and `india_*` assets
remain as source/alternate visuals.

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
| `capabl_india_microgrid_hero.webp` | `capabl_india_microgrid_hero.png` | Premium hero: blue-hour Indian society microgrid with rooftop solar, storage, EV charging, and text-safe negative space. |
| `capabl_rooftop_infrastructure.webp` | `capabl_rooftop_infrastructure.png` | Rooftop/courtyard infrastructure view showing solar, battery, transformer, and EV charging. |
| `capabl_society_operator.webp` | `capabl_society_operator.png` | Society manager and solar installer using a practical intelligence layer. |
| `capabl_neighbourhood_dispatch.webp` | `capabl_neighbourhood_dispatch.png` | Neighbourhood microgrid dispatch under evening grid stress. |
| `capabl_energy_journey_infographic.svg` | n/a | Deterministic horizontal infographic showing the journey from India's solar scale to local microgrid intelligence and community outcomes. |

These are editorial visuals for storytelling. For exact metrics and evidence, use the committed eval plots in `evals/plots/`.
