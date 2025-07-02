from argparse import Namespace
from pathlib import Path
from pdf2mp3 import cli
from unittest.mock import patch

FIXTURE_PDF = Path(__file__).parent / "fixture" / "hyperion.pdf"


class TestGetArgs:
    def test_default_args(self, monkeypatch):
        test_args = [
            str(FIXTURE_PDF),
        ]
        monkeypatch.setattr("sys.argv", ["prog"] + test_args)
        monkeypatch.setattr(Path, "is_file", lambda self: self == FIXTURE_PDF)
        args = cli.get_args()
        assert isinstance(args, Namespace)
        assert args.input_pdf == FIXTURE_PDF
        assert args.output_mp3 is None or args.output_mp3 == Path.cwd() / (
            FIXTURE_PDF.stem + ".mp3"
        )
        assert args.lang == "b"
        assert args.voice == "bf_emma"
        assert args.speed == 0.8
        assert args.split_pattern == r"[.”]\s*\n"
        assert args.bitrate == "CONSTANT"
        assert args.compression == 0.5

    def test_all_args(self, monkeypatch):
        test_args = [
            str(FIXTURE_PDF),
            "output.mp3",
            "-l",
            "a",
            "-v",
            "am_john",
            "-s",
            "1.2",
            "--split-pattern",
            r"[.!?] ",
            "--bitrate",
            "variable",
            "--compression",
            "0.8",
        ]
        monkeypatch.setattr("sys.argv", ["prog"] + test_args)
        monkeypatch.setattr(Path, "is_file", lambda self: self == FIXTURE_PDF)
        args = cli.get_args()
        assert args.input_pdf == FIXTURE_PDF
        assert args.output_mp3 == Path("output.mp3")
        assert args.lang == "a"
        assert args.voice == "am_john"
        assert args.speed == 1.2
        assert args.split_pattern == r"[.!?] "
        assert args.bitrate == "VARIABLE"
        assert args.compression == 0.8


class TestMain:
    def test_main_runs(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "get_args",
            lambda: Namespace(
                input_pdf=FIXTURE_PDF,
                output_mp3=Path("output.mp3"),
                lang="b",
                voice="bf_emma",
                speed=0.8,
                split_pattern=r"[.”]\s*\n",
                bitrate="CONSTANT",
                compression=0.5,
                device=None,
                show_progress=True,
                overwrite=False,
            ),
        )
        with patch("pdf2mp3.core.convert_pdf_to_mp3") as mock_convert:
            try:
                cli.main()
            except SystemExit as e:
                # Acceptable if main() calls sys.exit
                assert e.code in (0, 1)
            mock_convert.assert_called_once()
            called_kwargs = mock_convert.call_args.kwargs
            assert called_kwargs["pdf_path"] == FIXTURE_PDF
            assert called_kwargs["output_mp3_path"] == Path("output.mp3")
            assert called_kwargs["lang"] == "b"
            assert called_kwargs["voice"] == "bf_emma"
            assert called_kwargs["speed"] == 0.8
            assert called_kwargs["split_pattern"] == r"[.”]\s*\n"
            assert called_kwargs["bitrate_mode"] == "CONSTANT"
            assert called_kwargs["compression_level"] == 0.5
            assert called_kwargs["device"] is None
            assert called_kwargs["show_progress"] is True
            assert called_kwargs["overwrite"] is False

    def test_main_runs_with_custom_split_pattern(self, monkeypatch):
        custom_pattern = r"[.]\s*\n"
        monkeypatch.setattr(
            cli,
            "get_args",
            lambda: Namespace(
                input_pdf=FIXTURE_PDF,
                output_mp3=Path("output.mp3"),
                lang="b",
                voice="bf_emma",
                speed=0.8,
                split_pattern=custom_pattern,
                bitrate="CONSTANT",
                compression=0.5,
                device=None,
                show_progress=True,
                overwrite=False,
            ),
        )
        with patch("pdf2mp3.core.convert_pdf_to_mp3") as mock_convert:
            try:
                cli.main()
            except SystemExit as e:
                assert e.code in (0, 1)
            mock_convert.assert_called_once()
            called_kwargs = mock_convert.call_args.kwargs
            assert called_kwargs["split_pattern"] == custom_pattern
