import pytest
from pathlib import Path
from pdf2mp3 import core


FIXTURE_PDF = Path(__file__).parent / "fixture" / "hyperion.pdf"


class TestExtractTextFromPDF:
    def test_extract_text(self):
        text = core.extract_text_from_pdf(FIXTURE_PDF)
        assert isinstance(text, str)
        assert len(text) > 1000  # Should be non-trivial text
        # Check for a phrase known to be in the first page, or just that text is not empty
        assert any(
            word in text for word in ["Consul", "Gladstone", "fatline", "transmission"]
        )


class TestConvertPDFToMP3:
    def test_real_synth_and_size(self, tmp_path):
        output_mp3 = tmp_path / "hyperion.mp3"
        # Remove if exists
        if output_mp3.exists():
            output_mp3.unlink()
        core.convert_pdf_to_mp3(
            pdf_path=FIXTURE_PDF,
            output_mp3_path=output_mp3,
            lang="b",
            voice="bf_emma",
            speed=0.8,
            split_pattern=r"[.”]\s*\n",
            bitrate_mode="CONSTANT",
            compression_level=0.5,
            device=None,
            show_progress=False,
            overwrite=True,
        )
        assert output_mp3.exists()
        # Check file size is about 1:48 duration (empirically ~1084320 bytes, allow ±5%)
        size = output_mp3.stat().st_size
        assert 1030000 <= size <= 1140000, f"Unexpected mp3 size: {size}"
