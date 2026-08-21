"""CLI tests — config-default precedence for `pick`.

Omitted --intent/--dpi must fall back to the config file's [defaults],
and explicit options must override it.
"""

import numpy as np
from typer.testing import CliRunner

from photolab.cli import app

runner = CliRunner()


def _write_config(tmp_path, dpi: int, intent: str):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[defaults]\n"
        f"dpi = {dpi}\n"
        'paper = "matte"\n'
        f'intent = "{intent}"\n'
    )
    return cfg


class TestPickConfigDefaults:
    def test_config_defaults_used_when_options_omitted(
        self, sample_tiff_16bit, tmp_path, monkeypatch
    ):
        cfg_path = _write_config(tmp_path, 240, "relative-colorimetric")
        monkeypatch.setattr("photolab.config.config_path", lambda: cfg_path)

        result = runner.invoke(
            app,
            ["pick", str(sample_tiff_16bit), "--output-dir", str(tmp_path / "print")],
        )

        assert result.exit_code == 0, result.output
        assert "Intent: relative-colorimetric, DPI: 240" in result.output

    def test_explicit_options_override_config(
        self, sample_tiff_16bit, tmp_path, monkeypatch
    ):
        cfg_path = _write_config(tmp_path, 240, "relative-colorimetric")
        monkeypatch.setattr("photolab.config.config_path", lambda: cfg_path)

        result = runner.invoke(
            app,
            [
                "pick", str(sample_tiff_16bit),
                "--intent", "perceptual", "--dpi", "300",
                "--output-dir", str(tmp_path / "print"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Intent: perceptual, DPI: 300" in result.output


class TestPickSixteenBit:
    def test_16bit_survives_pick_with_profile(self, sample_tiff_16bit, tmp_path, monkeypatch):
        """End-to-end: 16-bit TIFF in -> pick with ICC profile -> 16-bit TIFF out.

        An 8-bit round-trip anywhere in the path (load, ICC transform, or the
        tag-attaching save) collapses the gradient to <=256 levels per channel.
        """
        import tifffile
        from PIL import ImageCms

        icc = tmp_path / "srgb.icc"
        icc.write_bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[defaults]\n"
            "dpi = 300\n"
            'paper = "matte"\n'
            'intent = "perceptual"\n'
            "\n"
            "[profiles.matte]\n"
            f'path = "{icc}"\n'
            'type = "matte"\n'
        )
        monkeypatch.setattr("photolab.config.config_path", lambda: cfg)
        out_dir = tmp_path / "print"

        result = runner.invoke(app, ["pick", str(sample_tiff_16bit), "--output-dir", str(out_dir)])
        assert result.exit_code == 0, result.output

        out_path = out_dir / "sample_print.tiff"
        arr = tifffile.imread(str(out_path))
        assert arr.dtype == np.uint16
        for ch in range(3):
            assert len(np.unique(arr[..., ch])) > 256
        with tifffile.TiffFile(str(out_path)) as tf:
            tags = tf.pages[0].tags
            assert 34675 in tags  # ICC profile attached
            assert tags["XResolution"].value[0] / tags["XResolution"].value[1] == 300


class TestRefineRecipeIdValidation:
    def test_traversal_recipe_id_rejected(self, sample_jpeg, tmp_path):
        """recipe_id lands in the output filename — a traversal payload must be
        rejected before anything is written."""
        import json

        prescription = tmp_path / "rx.json"
        prescription.write_text(json.dumps({
            "diagnostic": "d",
            "recipes": [{
                "recipe_id": "../../../../tmp/x",
                "base_variant": "v1_as_shot",
                "adjustments": [],
            }],
        }))
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app, ["refine", str(sample_jpeg), str(prescription), "--output-dir", str(out_dir)]
        )

        assert result.exit_code == 0, result.output
        assert "0 recipe(s) applied, 1 skipped" in result.output
        # Nothing was written anywhere under the test tree
        assert list(tmp_path.rglob("*.tiff")) == []

    def test_valid_recipe_id_still_accepted(self, sample_jpeg, tmp_path):
        import json

        prescription = tmp_path / "rx.json"
        prescription.write_text(json.dumps({
            "diagnostic": "d",
            "recipes": [{
                "recipe_id": "R1",
                "base_variant": "v1_as_shot",
                "adjustments": [],
            }],
        }))
        out_dir = tmp_path / "out"

        result = runner.invoke(
            app, ["refine", str(sample_jpeg), str(prescription), "--output-dir", str(out_dir)]
        )

        assert result.exit_code == 0, result.output
        assert (out_dir / "sample_R1.tiff").exists()
