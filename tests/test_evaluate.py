import json
import pytest
from photolab.evaluate import parse_evaluation_response


SAMPLE_RESPONSE = """## DIAGNOSTIC

**V1 — As Shot**: Slightly warm overall with good exposure on the subjects. The sky is flat but retains some cloud detail.

**V2 — Auto Levels**: Improved contrast and color separation. Skin tones are natural. Best overall balance.

**V5 — CLAHE**: Strong local contrast brings out texture in clothing and sand, but skin looks slightly harsh.

**V6 — Warm +500K**: Pleasant warmth to skin tones, but the sand and sky take on an unnatural golden cast.

### Best-of assessment
- Best exposure: V8
- Best skin tone: V6
- Best local contrast: V5
- Best highlight preservation: V9
- Best shadow detail: V8

## PRESCRIPTION

```json
[
  {
    "recipe_id": "R1",
    "label": "Safe natural lift",
    "description": "Auto levels base with gentle exposure lift. Conservative and safe.",
    "base_variant": "v2_auto_levels",
    "adjustments": [
      {"type": "exposure", "value": 0.2, "strength": 1.0}
    ]
  },
  {
    "recipe_id": "R2",
    "label": "Warm lift, protected sky",
    "description": "Warm base with exposure lift, protecting highlights to keep sky detail.",
    "base_variant": "v6_warm",
    "adjustments": [
      {"type": "exposure", "value": 0.3, "strength": 0.8},
      {"type": "highlight_protection", "threshold": 0.75, "strength": 0.7}
    ]
  },
  {
    "recipe_id": "R3",
    "label": "Punchy split-tone",
    "description": "Auto levels with CLAHE in shadows/midtones only, warm shift, and highlight protection.",
    "base_variant": "v2_auto_levels",
    "adjustments": [
      {"type": "clahe", "strength": 0.4, "zone": "midtones"},
      {"type": "color_temp", "value": 200, "strength": 1.0},
      {"type": "highlight_protection", "threshold": 0.7, "strength": 0.8}
    ]
  }
]
```"""


class TestParseEvaluationResponse:
    def test_extracts_diagnostic(self):
        diagnostic, recipes = parse_evaluation_response(SAMPLE_RESPONSE)
        assert "V1" in diagnostic
        assert "Auto Levels" in diagnostic
        assert "Best exposure" in diagnostic

    def test_extracts_recipes(self):
        _, recipes = parse_evaluation_response(SAMPLE_RESPONSE)
        assert len(recipes) == 3
        assert recipes[0]["recipe_id"] == "R1"
        assert recipes[1]["base_variant"] == "v6_warm"
        assert recipes[2]["adjustments"][0]["zone"] == "midtones"

    def test_recipe_structure(self):
        _, recipes = parse_evaluation_response(SAMPLE_RESPONSE)
        for r in recipes:
            assert "recipe_id" in r
            assert "label" in r
            assert "base_variant" in r
            assert "adjustments" in r

    def test_no_json_raises(self):
        with pytest.raises(RuntimeError, match="No JSON"):
            parse_evaluation_response("Just a diagnostic, no prescription.")

    def test_diagnostic_excludes_json(self):
        diagnostic, _ = parse_evaluation_response(SAMPLE_RESPONSE)
        assert "```json" not in diagnostic
        assert "recipe_id" not in diagnostic
