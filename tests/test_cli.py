"""Tests for the pdf2mp3.cli module."""

import pytest
from pathlib import Path
import subprocess
from unittest import mock
import sys

# Attempt to import main from cli, skip if it causes ImportError due to missing dependencies during test collection
try:
    from pdf2mp3 import cli
except ImportError:
    cli = None

# Helper to get package version
def get_package_version():
    try:
        from pdf2mp3 import __version__ as v
        return v
    except ImportError:
        return "0.0.0" # Fallback

PKG_VERSION = get_package_version()

# Skip all tests in this module if cli could not be imported
pytestmark = pytest.mark.skipif(cli is None, reason="CLI module (or its dependencies like torch/TTS) not available for testing.")

@pytest.fixture
def mock_args(tmp_path, monkeypatch):
    """Fixture to mock sys.argv for testing argparse."""
    def _mock_args(arg_list):
        # Prepend program name (can be anything, as argparse ignores it)
        full_args = ["pdf2mp3_test"] + arg_list
        monkeypatch.setattr(sys, "argv", full_args)
    return _mock_args

@pytest.fixture
def dummy_pdf(tmp_path):
    """Creates a dummy PDF file for testing purposes."""
    pdf_path = tmp_path / "test_document.pdf"
    # Use bytes for a minimal valid PDF structure
    minimal_pdf_content = b"""%PDF-1.0
%BINARY
1 0 obj
  << /Type /Catalog
     /Pages 2 0 R
  >>
endobj
2 0 obj
  << /Type /Pages
     /Kids [3 0 R]
     /Count 1
  >>
endobj
3 0 obj
  << /Type /Page
     /Parent 2 0 R
     /MediaBox [0 0 612 792]
     /Contents 4 0 R
     /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
  >>
endobj
4 0 obj
  << /Length 19 >>
stream
BT /F1 12 Tf 100 100 Td ( ) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000016 00000 n
0000000070 00000 n
0000000160 00000 n
0000000290 00000 n
trailer
  << /Size 5
     /Root 1 0 R
  >>
%%EOF
"""
    pdf_path.write_bytes(minimal_pdf_content)
    return pdf_path

@pytest.fixture
def mock_convert_pdf_to_mp3(monkeypatch):
    """Mocks the core.convert_pdf_to_mp3 function by replacing the core module in sys.modules."""
    mock_converter_function = mock.Mock(name="convert_pdf_to_mp3_mock_function")

    # Create a mock core module object
    mock_core_module = mock.MagicMock(name="mock_core_module")
    mock_core_module.convert_pdf_to_mp3 = mock_converter_function

    # Before cli.main() is called (which contains the lazy 'from . import core'),
    # we insert our mock_core_module into sys.modules.
    # The key 'pdf2mp3.core' must match what the import system will look for.
    # When 'from . import core' is executed within 'pdf2mp3.cli', it should find this mock.
    monkeypatch.setitem(sys.modules, "pdf2mp3.core", mock_core_module)

    return mock_converter_function # Return the mock function itself for assertions

def run_cli_main_with_args(arg_list, monkeypatch):
    """Helper to run cli.main() with specified args and catch SystemExit."""
    # The first element of sys.argv is the program name. argparse uses this for help messages
    # if 'prog' is not set in ArgumentParser. Here, 'prog' is set to 'pdf2mp3'.
    # So, this first element "pdf2mp3_test_runner" doesn't affect the help output prog name.
    full_args = ["pdf2mp3_test_runner"] + arg_list
    monkeypatch.setattr(sys, "argv", full_args)
    try:
        cli.main()
    except SystemExit as e:
        return e.code # Return exit code
    return 0 # Default exit code for success

# --- Test Cases ---

def test_cli_invoked_no_args_shows_help(capsys, monkeypatch):
    """Test that invoking with no arguments shows help and exits."""
    # argparse by default prints help to stderr if required args are missing
    exit_code = run_cli_main_with_args([], monkeypatch)
    assert exit_code != 0 # Should exit with error due to missing arg
    captured = capsys.readouterr()
    assert "usage: pdf2mp3 [-h]" in captured.err # Actual prog name is 'pdf2mp3'
    assert "input_pdf" in captured.err # Missing required argument

def test_cli_version(capsys, monkeypatch):
    """Test the --version flag."""
    exit_code = run_cli_main_with_args(["--version"], monkeypatch)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"pdf2mp3 {PKG_VERSION}" in captured.out # Actual prog name is 'pdf2mp3'

def test_cli_help(capsys, monkeypatch):
    """Test the --help flag."""
    exit_code = run_cli_main_with_args(["--help"], monkeypatch)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage: pdf2mp3 [-h]" in captured.out # Actual prog name is 'pdf2mp3'
    assert "Convert a PDF e-book to a single MP3 file" in captured.out

def test_input_pdf_required(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    """Test that input_pdf is correctly passed."""
    mock_args([str(dummy_pdf)])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    call_args = mock_convert_pdf_to_mp3.call_args[1]
    assert call_args['pdf_path'] == dummy_pdf

def test_output_mp3_optional_default(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    """Test default output_mp3 path."""
    mock_args([str(dummy_pdf)])
    cli.main()
    expected_output_path = Path.cwd() / (dummy_pdf.stem + ".mp3")
    mock_convert_pdf_to_mp3.assert_called_once()
    call_args = mock_convert_pdf_to_mp3.call_args[1]
    assert call_args['output_mp3_path'] == expected_output_path

def test_output_mp3_custom(dummy_pdf, tmp_path, mock_convert_pdf_to_mp3, mock_args):
    """Test custom output_mp3 path."""
    custom_output = tmp_path / "custom_audio.mp3"
    mock_args([str(dummy_pdf), str(custom_output)])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    call_args = mock_convert_pdf_to_mp3.call_args[1]
    assert call_args['output_mp3_path'] == custom_output

# Test Core Synthesis Options
def test_lang_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--lang", "a"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['lang'] == "a"

def test_voice_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--voice", "test_voice"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['voice'] == "test_voice"

def test_speed_option_valid(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--speed", "1.5"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['speed'] == 1.5

def test_speed_option_invalid_low(dummy_pdf, capsys, monkeypatch):
    exit_code = run_cli_main_with_args([str(dummy_pdf), "--speed", "0.4"], monkeypatch)
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Speed must be between 0.5 and 2.0" in captured.err

def test_speed_option_invalid_high(dummy_pdf, capsys, monkeypatch):
    exit_code = run_cli_main_with_args([str(dummy_pdf), "--speed", "2.1"], monkeypatch)
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Speed must be between 0.5 and 2.0" in captured.err

def test_split_pattern_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--split-pattern", "\\n\\n"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['split_pattern'] == "\\n\\n"

# Test Audio Encoding Options
def test_bitrate_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--bitrate", "VARIABLE"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['bitrate_mode'] == "VARIABLE"

def test_compression_option_valid(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--compression", "0.7"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['compression_level'] == 0.7

def test_compression_option_invalid_low(dummy_pdf, capsys, monkeypatch):
    exit_code = run_cli_main_with_args([str(dummy_pdf), "--compression", "-0.1"], monkeypatch)
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Compression level must be between 0.0 and 1.0" in captured.err

def test_compression_option_invalid_high(dummy_pdf, capsys, monkeypatch):
    exit_code = run_cli_main_with_args([str(dummy_pdf), "--compression", "1.1"], monkeypatch)
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "Compression level must be between 0.0 and 1.0" in captured.err

# Test Runtime and I/O Options
def test_device_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--device", "cuda:1"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['device'] == "cuda:1"

def test_overwrite_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--overwrite"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['overwrite'] is True

# Removed test_resume_option as --resume CLI argument is deleted.

# Removed test_tmp_dir_option as --tmp-dir CLI argument is deleted.

def test_no_progress_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    mock_args([str(dummy_pdf), "--no-progress"])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['show_progress'] is False

def test_default_progress_option(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    """Test that progress is shown by default."""
    mock_args([str(dummy_pdf)])
    cli.main()
    mock_convert_pdf_to_mp3.assert_called_once()
    assert mock_convert_pdf_to_mp3.call_args[1]['show_progress'] is True

# Test Argument Validation
def test_input_pdf_non_existent(tmp_path, capsys, monkeypatch):
    non_existent_pdf = tmp_path / "does_not_exist.pdf"
    exit_code = run_cli_main_with_args([str(non_existent_pdf)], monkeypatch)
    assert exit_code != 0
    captured = capsys.readouterr()
    assert f"Input PDF file not found: {non_existent_pdf}" in captured.err

# Note: Corrected to use the mock_convert_pdf_to_mp3 fixture properly.
def test_core_convert_exception_handling(dummy_pdf, mock_args, monkeypatch, capsys, mock_convert_pdf_to_mp3):
    """Test that exceptions from core.convert_pdf_to_mp3 are caught and handled."""
    # mock_convert_pdf_to_mp3 is the mock function provided by the fixture.
    # Configure its side_effect for this specific test.
    mock_convert_pdf_to_mp3.side_effect = Exception("Core processing failed!")

    # mock_args is used by run_cli_main_with_args implicitly through monkeypatch if not passed explicitly to cli.main
    # We need to ensure sys.argv is set up by mock_args before calling run_cli_main_with_args,
    # or rely on run_cli_main_with_args's internal monkeypatching.
    # The current run_cli_main_with_args handles monkeypatching sys.argv itself.

    exit_code = run_cli_main_with_args([str(dummy_pdf)], monkeypatch)

    assert exit_code == 1 # Should exit with 1 on error
    captured = capsys.readouterr()
    assert "An unexpected error occurred during conversion: Core processing failed!" in captured.err
    mock_convert_pdf_to_mp3.assert_called_once() # Assert on the correct mock

def test_core_import_error_handling(dummy_pdf, mock_args, monkeypatch, capsys):
    """Test that ImportError for the core module is handled."""

    # To simulate an ImportError for 'from . import core' within cli.main,
    # we can make 'pdf2mp3.core' temporarily un-importable.
    # This is a bit tricky. One way is to ensure 'pdf2mp3.core' is not in sys.modules
    # and that trying to import it fails.
    # For this test, we'll simulate that 'from . import core' fails.
    # The most robust way to do this is to make 'pdf2mp3.core' un-importable
    # by monkeypatching __import__ or by removing 'pdf2mp3.core' from sys.modules
    # and ensuring it cannot be found.

    # We will use monkeypatch to temporarily replace the __import__ built-in
    original_import = __builtins__["__import__"] # Store the original

    def mocked_import(name, globals_dict=None, locals_dict=None, fromlist=(), level=0):
        # Check if the import is for 'core' relative to 'pdf2mp3.cli'
        # The `name` will be 'core', `fromlist` will be non-empty, and `level` will be 1 for 'from . import core'
        # `globals_dict['__name__']` should be 'pdf2mp3.cli'
        if name == "core" and globals_dict and globals_dict.get("__name__") == "pdf2mp3.cli" and level == 1:
            raise ImportError("Simulated core import error for test")
        elif name == "pdf2mp3.core": # Also catch direct import if that path is taken
             raise ImportError("Simulated core import error for test")
        return original_import(name, globals_dict, locals_dict, fromlist, level)

    # Patch the __import__ in the 'builtins' module
    monkeypatch.setattr(sys.modules['builtins'], "__import__", mocked_import)

    # It's also good practice to ensure that pdf2mp3.core is not already in sys.modules from a previous import.
    # The mock_convert_pdf_to_mp3 fixture might put a mock there, so we ensure it's removed
    # for this specific test if we want to test the ImportError path cleanly.
    if "pdf2mp3.core" in sys.modules:
        monkeypatch.delitem(sys.modules, "pdf2mp3.core", raising=False)

    exit_code = run_cli_main_with_args([str(dummy_pdf)], monkeypatch)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Could not import the 'core' module." in captured.err

# It might be good to also test default values for all options
# if they are not explicitly tested by being passed to the mock.
# The mock_convert_pdf_to_mp3.call_args[1] contains all keyword args,
# so we can check their default values there if not specified.

def test_all_defaults_passed_to_core(dummy_pdf, mock_convert_pdf_to_mp3, mock_args):
    """Test that all arguments have correct default values passed to core function."""
    mock_args([str(dummy_pdf)])
    cli.main()

    mock_convert_pdf_to_mp3.assert_called_once()
    kwargs = mock_convert_pdf_to_mp3.call_args[1]

    assert kwargs['pdf_path'] == dummy_pdf
    assert kwargs['output_mp3_path'] == Path.cwd() / (dummy_pdf.stem + ".mp3")
    assert kwargs['lang'] == "b"
    assert kwargs['voice'] == "bf_emma"
    assert kwargs['speed'] == 0.8
    assert kwargs['split_pattern'] == r'[.”]\s*\n'
    assert kwargs['bitrate_mode'] == 'CONSTANT'
    assert kwargs['compression_level'] == 0.5
    assert kwargs['device'] is None # core.py handles auto-detect
    # tmp_dir is no longer a parameter for core.convert_pdf_to_mp3
    # resume is no longer a parameter for core.convert_pdf_to_mp3
    assert 'tmp_dir' not in kwargs
    assert 'resume' not in kwargs
    assert kwargs['show_progress'] is True
    assert kwargs['overwrite'] is False

# Example of how to use subprocess for a very basic CLI invocation test
# This is more of an integration test and might be slower / more complex to set up.
# For unit testing argument parsing, mocking sys.argv and calling main() is usually sufficient.
@pytest.mark.skipif(cli is None, reason="CLI module not available for subprocess test.")
def test_cli_with_subprocess_version(monkeypatch):
    """Test CLI invocation via subprocess for --version. Requires package to be installed."""
    # This test assumes that `pip install -e .` or `pip install .` has been run,
    # making the `pdf2mp3` script available on the PATH within the test environment.
    try:
        result = subprocess.run(["pdf2mp3", "--version"], capture_output=True, text=True, check=True)
        assert result.returncode == 0
        assert f"pdf2mp3 {PKG_VERSION}" in result.stdout.strip()
    except FileNotFoundError:
        pytest.skip("pdf2mp3 command not found in PATH. Run 'pip install -e .' first.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI subprocess call failed: {e}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")


# @pytest.mark.skip(reason="Subprocess tests are more like integration tests, slower")
# def test_cli_with_subprocess_help():
# """Test CLI invocation via subprocess for --help."""
# result = subprocess.run([sys.executable, "-m", "pdf2mp3.cli", "--help"], capture_output=True, text=True)
# assert result.returncode == 0
# assert "usage: cli.py [-h]" in result.stdout # Note: prog name might be cli.py
# assert "Convert a PDF e-book" in result.stdout

# Final check: ensure cli.main is callable if not skipped
if cli is not None:
    assert callable(cli.main)

# TODO: Consider tests for interactions, e.g., --resume without a tmp_dir,
# or --overwrite with an existing file (though this is more core logic behavior).
# For CLI tests, we mostly care that the flag is correctly parsed and passed.

# TODO: Test for invalid choices for arguments with 'choices' like --bitrate
def test_bitrate_option_invalid_choice(dummy_pdf, capsys, monkeypatch):
    exit_code = run_cli_main_with_args([str(dummy_pdf), "--bitrate", "INVALID_CHOICE"], monkeypatch)
    assert exit_code != 0
    captured = capsys.readouterr()
    assert "invalid choice: 'INVALID_CHOICE'" in captured.err # argparse default error
    assert "(choose from 'CONSTANT', 'VARIABLE')" in captured.err
