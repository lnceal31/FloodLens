"""FloodLens Vision (prototype).

A lightweight, dependency-free (NumPy + Pillow) flood-surface analyser. It
segments water-like surfaces in the ground region of a photograph using colour
and local-texture cues, then converts the covered fraction into a visual
severity category.

This is a heuristic prototype, NOT a trained neural network. A production
deployment would replace ``analyze_image`` with a segmentation model fine-tuned
on labelled flood imagery while keeping the same return structure.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

CLASSES = {
    "none": "Moderate flooding",
    "low": "Low visible flooding",
    "moderate": "Moderate visible flooding",
    "severe": "Severe visible flooding",
}
# Mapping to the severity scale used for reports.
REPORT_SEVERITY = {"none": None, "low": "Low", "moderate": "Medium", "severe": "High"}

THRESHOLDS = (0.06, 0.18, 0.38)
GROUND_START = 0.38  # ignore the top part of the frame (sky / skyline)
BLOCK = 8


def _explain(key: str, pct: int) -> str:
    if key == "none":
        return (
            "No significant standing-water signature was detected. Surfaces in the lower part of "
            "the frame show normal texture and colour variation."
        )
    if key == "low":
        return (
            f"Small patches of smooth, reflective surface cover roughly {pct}% of the ground area, "
            "consistent with shallow ponding. The road is likely passable but worth monitoring."
        )
    if key == "moderate":
        return (
            f"A continuous water-like surface covers about {pct}% of the ground area, suggesting "
            "standing water across part of the carriageway. Traffic may be affected and drainage "
            "inspection is recommended."
        )
    return (
        f"Water-like surface dominates roughly {pct}% of the ground area, consistent with widespread "
        "flooding. This location should be prioritised for response and on-site verification."
    )


def _analyze_heuristic(image: Image.Image) -> dict:
    img = ImageOps.exif_transpose(image).convert("RGB")
    img.thumbnail((384, 384))
    if min(img.size) < 32:
        raise ValueError("Image is too small to analyse (minimum 32 px on each side).")

    w, h = img.size
    hh, ww = (h // BLOCK) * BLOCK, (w // BLOCK) * BLOCK
    img = img.crop((0, 0, ww, hh))

    hsv = np.asarray(img.convert("HSV")).astype(np.float32)
    hue = hsv[..., 0] * 360.0 / 255.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    gray = np.asarray(ImageOps.grayscale(img)).astype(np.float32) / 255.0

    bh, bw = hh // BLOCK, ww // BLOCK

    def blocks(a: np.ndarray) -> np.ndarray:
        return a.reshape(bh, BLOCK, bw, BLOCK)

    tex = blocks(gray).std(axis=(1, 3))
    hue_b = blocks(hue).mean(axis=(1, 3))
    sat_b = blocks(sat).mean(axis=(1, 3))
    val_b = blocks(val).mean(axis=(1, 3))

    smooth = tex < 0.045
    blue_water = (hue_b >= 165) & (hue_b <= 255) & (sat_b > 0.08) & (val_b > 0.25) & smooth
    muddy_water = (hue_b >= 15) & (hue_b <= 55) & (sat_b > 0.18) & (sat_b < 0.70) & (val_b > 0.20) & (tex < 0.03)
    reflective = (sat_b < 0.12) & (val_b > 0.55) & smooth

    candidate = blue_water | muddy_water | reflective
    ground = np.zeros_like(candidate)
    ground[int(bh * GROUND_START):, :] = True
    candidate &= ground

    # Remove isolated speckle.
    mask_img = Image.fromarray((candidate * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3))
    candidate = np.asarray(mask_img) > 127
    coverage = float(candidate.sum() / max(ground.sum(), 1))

    if coverage < THRESHOLDS[0]:
        key = "none"
    elif coverage < THRESHOLDS[1]:
        key = "low"
    elif coverage < THRESHOLDS[2]:
        key = "moderate"
    else:
        key = "severe"

    margin = min(abs(coverage - t) for t in THRESHOLDS)
    confidence = 0.62 + min(margin / 0.12, 1.0) * 0.30
    if key == "none" and coverage < 0.02:
        confidence = max(confidence, 0.88)
    confidence = float(min(confidence, 0.93))

    overlay_mask = mask_img.resize(img.size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(2))
    overlay_mask = overlay_mask.point(lambda p: int(p * 0.55))
    tint = Image.new("RGB", img.size, (32, 197, 216))
    overlay = Image.composite(tint, img, overlay_mask)

    return {
        "key": key,
        "label": CLASSES[key],
        "severity": REPORT_SEVERITY[key],
        "coverage": coverage,
        "confidence": confidence,
        "explanation": _explain(key, int(round(coverage * 100))),
        "original": img,
        "overlay": overlay,
        "engine": "Prototype (colour and texture analysis)",
    }


# ---------------------------------------------------------------------------
# AI engine (vision-language model via API)
# ---------------------------------------------------------------------------
AI_MODEL = "claude-sonnet-5-5"
AI_PROMPT = """You are the flood-assessment engine of FloodLens, a flood intelligence platform.
Look at this photograph of a road or urban area and judge ONLY what is visibly shown.

Classify visible flooding as exactly one of:
- "none": no standing or flowing flood water visible
- "low": shallow puddles or minor ponding, road clearly passable
- "moderate": water covering part of the carriageway or reaching vehicle wheels
- "severe": deep or widespread water, submerged vehicles or road, evident danger

Reply with a single JSON object and nothing else:
{"key": "none|low|moderate|severe", "confidence": <number 0 to 1>, "explanation": "<one or two plain sentences describing what you can see that supports the classification>"}

If the image is not of a road or outdoor area, use "none" and say so in the explanation."""


def _extract_json(text: str) -> dict:
    import json

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model reply")
    return json.loads(text[start : end + 1])


def _analyze_ai(image: Image.Image, api_key: str) -> dict:
    import base64
    import io

    import anthropic

    img = ImageOps.exif_transpose(image).convert("RGB")
    img.thumbnail((1024, 1024))
    if min(img.size) < 32:
        raise ValueError("Image is too small to analyse (minimum 32 px on each side).")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = anthropic.Anthropic(api_key=api_key, timeout=25.0, max_retries=1)
    msg = client.messages.create(
        model=AI_MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": AI_PROMPT},
                ],
            }
        ],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    out = _extract_json(text)

    key = str(out.get("key", "")).lower().strip()
    if key not in CLASSES:
        raise ValueError(f"Unexpected classification: {key!r}")
    confidence = max(0.0, min(float(out.get("confidence", 0.8)), 1.0))
    explanation = str(out.get("explanation", "")).strip() or _explain(key, 0)

    return {
        "key": key,
        "label": CLASSES[key],
        "severity": REPORT_SEVERITY[key],
        "coverage": None,
        "confidence": confidence,
        "explanation": explanation,
        "original": img,
        "overlay": None,
        "engine": "FloodLens Vision AI",
    }



def analyze_image(image: Image.Image) -> dict:
    """Analyse an image using Anthropic AI, with a local fallback."""

    import os

    if min(image.size) < 32:
        raise ValueError(
            "Image is too small to analyse (minimum 32 px on each side)."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        print("FLOODLENS WARNING: No Anthropic API key found.")
        return _analyze_heuristic(image)

    try:
        result = _analyze_ai(image, api_key)
        print("FLOODLENS: Anthropic AI analysis successful.")
        return result

    except Exception as e:
        print("FLOODLENS AI ERROR:")
        print(type(e).__name__, str(e))
        print("Switching to local prototype.")
        return _analyze_heuristic(image)
