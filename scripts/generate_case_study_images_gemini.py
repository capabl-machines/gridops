"""Generate Capabl/GridOps case-study images with Gemini image models.

The script reads `GEMINI_API_KEY` or `GEMINI_API_KEY2` from the environment and
writes generated PNG assets into `assets/case_study/`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from google import genai
from google.genai import types


ASSETS = [
    {
        "slug": "gridops_hero_microgrid",
        "prompt": (
            "A cinematic but realistic editorial hero image for a technical case study about AI-operated "
            "community microgrids in India. Rooftop solar panels, a neighborhood battery container, a small "
            "diesel backup unit, power lines, and homes at evening blue hour. Subtle visual hints of software "
            "control and energy flow, no readable text, no logos, sophisticated enterprise technology style, "
            "natural colors, high detail, 16:9."
        ),
    },
    {
        "slug": "gridops_control_room",
        "prompt": (
            "A modern grid operations control room focused on community microgrid dispatch. Engineers view "
            "battery state, solar generation, outage alerts, and cost/reliability charts on large screens. "
            "Human-scale, credible, non-futuristic, warm professional lighting, no readable text, no brand logos, "
            "photo-realistic editorial image, 16:9."
        ),
    },
    {
        "slug": "gridops_environment_loop",
        "prompt": (
            "An elegant technical visual metaphor for an AI decision loop controlling a microgrid: observation "
            "signals flow from solar, demand, grid price, battery, and diesel into a compact model, then JSON-like "
            "action signals flow back to the battery, diesel, and demand response. Clean isometric information "
            "design, minimal labels or no readable text, dark neutral background with precise colored energy "
            "flows, enterprise case-study style, 16:9."
        ),
    },
    {
        "slug": "gridops_impact_split",
        "prompt": (
            "A split-scene editorial illustration showing impact of AI microgrid dispatch. Left side: do-nothing "
            "operation with dim homes during a heatwave outage and stressed grid. Right side: trained model "
            "operation with battery dispatch, fewer blackouts, stable homes, and cleaner energy flow. Realistic "
            "but polished, Indian neighborhood context, no readable text, no logos, 16:9."
        ),
    },
]


def save_first_image(response, output_path: Path) -> bool:
    """Save the first image part from a Gemini generate_content response."""
    for candidate in response.candidates or []:
        content = candidate.content
        if not content:
            continue
        for part in content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                output_path.write_bytes(inline.data)
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="assets/case_study")
    parser.add_argument("--model", default=os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"))
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--image-size", default="2K")
    parser.add_argument("--only", default="", help="Comma-separated asset slugs to generate.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY2")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GEMINI_API_KEY2.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    client = genai.Client(api_key=api_key)

    selected = [asset for asset in ASSETS if not wanted or asset["slug"] in wanted]
    for asset in selected:
        output_path = output_dir / f"{asset['slug']}.png"
        response = client.models.generate_content(
            model=args.model,
            contents=[asset["prompt"]],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=args.aspect_ratio,
                    image_size=args.image_size,
                ),
            ),
        )
        if not save_first_image(response, output_path):
            raise RuntimeError(f"No image returned for {asset['slug']}")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
