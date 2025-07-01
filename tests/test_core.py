"""Tests for core pdf2mp3 functionality."""

import pytest
from pathlib import Path
from unittest import mock

# Attempt to import the functions to be tested
# This will allow early feedback if the path or module name is incorrect.
try:
    from pdf2mp3.core import extract_text_from_pdf, get_device
    import PyPDF2 # For mocking PdfReadError
except ImportError as e:
    # If there's an issue with imports, it's critical to know.
    # This could be due to PYTHONPATH issues in the test environment
    # or incorrect module structure.
    pytest.fail(f"Failed to import from pdf2mp3.core: {e}. Check PYTHONPATH and module structure.")


@pytest.fixture
def fixture_dir():
    """Returns the path to the test fixture directory."""
    return Path(__file__).parent / "fixture"

@pytest.fixture
def hyperion_pdf_path(fixture_dir):
    """Path to the hyperion.pdf test file."""
    return fixture_dir / "hyperion.pdf"

@pytest.fixture
def empty_pdf_path(tmp_path):
    """Creates a dummy empty PDF file for testing."""
    # PyPDF2 cannot read an empty file as a PDF, it will raise PdfReadError.
    # To test the "no text extracted" scenario for a valid but empty PDF,
    # we need a PDF with no text content.
    # For now, let's create a file that PyPDF2 can open but has no pages or no text.
    # A truly empty file will be caught by PdfReadError.
    # Let's create a file that is not a PDF to trigger a PdfReadError for one of the tests.
    p = tmp_path / "empty.pdf"
    p.write_text("This is not a PDF.") # This will cause PdfReadError
    return p

@pytest.fixture
def truly_empty_pdf_path(tmp_path):
    """Creates a PDF with no pages, which PyPDF2 can read but has no text."""
    p = tmp_path / "truly_empty.pdf"
    # Create a minimal valid PDF with no pages using PyPDF2
    # This is a bit tricky as PyPDF2 is mostly for reading.
    # Let's simulate the condition where pdf_reader.pages is empty.
    # For the actual test, we'll mock `PdfReader(fh).pages` to be empty.
    # The file itself can be minimal.
    p.write_bytes(b"%PDF-1.0\n%EOF\n") # A very minimal PDF, likely unreadable by PyPDF2 as having pages.
    return p


class TestExtractTextFromPdf:
    """Tests for the extract_text_from_pdf function."""

    def test_extract_with_hyperion_pdf_pattern1(self, hyperion_pdf_path):
        """Tests text extraction from hyperion.pdf with pattern r'[.”]\\s*\\n'."""
        chunks = extract_text_from_pdf(hyperion_pdf_path, r'[.”]\s*\n')
        assert len(chunks) == 5
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(chunk.strip() == chunk for chunk in chunks) # Ensure stripped

    def test_extract_with_hyperion_pdf_pattern2(self, hyperion_pdf_path):
        """Tests text extraction from hyperion.pdf with pattern r'[.]\\s*\\n'."""
        chunks = extract_text_from_pdf(hyperion_pdf_path, r'[.]\s*\n')
        assert len(chunks) == 4
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_extract_with_hyperion_pdf_newline_pattern(self, hyperion_pdf_path):
        """Tests text extraction from hyperion.pdf with simple newline r'\\n'."""
        chunks = extract_text_from_pdf(hyperion_pdf_path, r'\n')
        # Based on the description, expecting 31 chunks.
        # The actual number might vary slightly based on how PyPDF2 extracts text and handles multiple newlines.
        # Let's be somewhat flexible or adjust after first run if necessary.
        # For now, using the provided number.
        assert len(chunks) == 31
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_file_not_found(self, tmp_path, capsys):
        """Tests behavior when the PDF file is not found."""
        non_existent_pdf = tmp_path / "non_existent.pdf"
        chunks = extract_text_from_pdf(non_existent_pdf, r'\n')
        assert chunks == tuple()
        captured = capsys.readouterr()
        assert f"Error: PDF file not found at {non_existent_pdf}" in captured.err

    def test_empty_pdf_no_text_extraction(self, truly_empty_pdf_path, capsys):
        """
        Tests behavior with a PDF file that PyPDF2 can open but contains no actual text or pages.
        This simulates when `pdf_reader.pages` is empty or pages have no text.
        """
        # We need to ensure PyPDF2 opens it, but finds no pages or text.
        # The current `truly_empty_pdf_path` might cause PdfReadError.
        # A better way is to mock the behavior of PdfReader.

        mock_pdf_reader_instance = mock.Mock()
        mock_pdf_reader_instance.pages = [] # Simulate no pages

        with mock.patch('PyPDF2.PdfReader', return_value=mock_pdf_reader_instance) as mock_pdf_reader_class:
            # The file content for truly_empty_pdf_path doesn't strictly matter here as PdfReader is mocked,
            # but it needs to exist for the `open()` call.
            # We use truly_empty_pdf_path which creates a minimal file.
            chunks = extract_text_from_pdf(truly_empty_pdf_path, r'\n')
            assert chunks == tuple()
            mock_pdf_reader_class.assert_called_once() # Ensure PdfReader was called
            # Check if any error message was printed (should not be, for cleanly empty PDF)
            captured = capsys.readouterr()
            assert "Error" not in captured.out
            assert "Warning" not in captured.out

    def test_corrupted_pdf_read_error(self, empty_pdf_path, capsys):
        """
        Tests behavior when PyPDF2 encounters an error reading the PDF.
        The `empty_pdf_path` fixture creates a non-PDF file with a .pdf extension.
        """
        chunks = extract_text_from_pdf(empty_pdf_path, r'\n')
        assert chunks == tuple()
        captured = capsys.readouterr()
        # Check that PyPDF2's error is mentioned (or a generic one from our func)
        assert f"Error reading PDF file {empty_pdf_path}" in captured.err

    def test_pdf_read_error_mocked(self, tmp_path, capsys):
        """
        Tests behavior when PyPDF2.PdfReader raises PdfReadError more directly.
        """
        pdf_path = tmp_path / "raises_error.pdf"
        pdf_path.write_text("content") # File needs to exist

        with mock.patch('PyPDF2.PdfReader', side_effect=PyPDF2.errors.PdfReadError("Mocked PDF read error")):
            chunks = extract_text_from_pdf(pdf_path, r'\n')
            assert chunks == tuple()
            captured = capsys.readouterr()
            assert f"Error reading PDF file {pdf_path}: Mocked PDF read error" in captured.err


class TestGetDevice:
    """Tests for the get_device function."""

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_cuda_available_and_requested(self, mock_is_available):
        """Test get_device when CUDA is available and requested."""
        assert get_device() == "cuda"
        assert get_device("cuda") == "cuda"
        assert get_device("cuda:0") == "cuda:0"
        assert get_device("cuda:1") == "cuda:1"
        mock_is_available.assert_called() # Called for get_device() and get_device("cuda")

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_cuda_available_cpu_requested(self, mock_is_available):
        """Test get_device when CUDA is available but CPU is requested."""
        assert get_device("cpu") == "cpu"
        # torch.cuda.is_available should not be called if "cpu" is explicitly passed
        mock_is_available.assert_not_called()

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_cuda_not_available(self, mock_is_available, capsys):
        """Test get_device when CUDA is not available."""
        assert get_device() == "cpu" # Defaults to CPU
        mock_is_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_cuda_not_available_cuda_requested(self, mock_is_available, capsys):
        """Test get_device when CUDA is not available but was requested."""
        assert get_device("cuda") == "cpu" # Falls back to CPU
        captured = capsys.readouterr()
        assert "Warning: CUDA device 'cuda' requested but not available. Falling back to CPU." in captured.err
        mock_is_available.assert_called_once() # Called to check availability

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_cuda_not_available_specific_cuda_requested(self, mock_is_available, capsys):
        """Test get_device when specific CUDA device (e.g. cuda:0) is not available."""
        assert get_device("cuda:0") == "cpu" # Falls back to CPU
        captured = capsys.readouterr()
        assert "Warning: CUDA device 'cuda:0' requested but not available. Falling back to CPU." in captured.err
        mock_is_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_cuda_not_available_cpu_requested(self, mock_is_available):
        """Test get_device when CUDA is not available and CPU is requested."""
        assert get_device("cpu") == "cpu"
        # torch.cuda.is_available should not be called if "cpu" is explicitly passed
        mock_is_available.assert_not_called()

    def test_other_device_string_passthrough(self):
        """Test that arbitrary device strings are passed through if not 'cuda*'."""
        # This behavior is implicit: if not 'cuda' and not None, it's returned.
        # No mocking needed as torch.cuda.is_available isn't called for these.
        with mock.patch('torch.cuda.is_available') as mock_is_available:
            assert get_device("mps") == "mps"
            assert get_device("tpu") == "tpu"
            mock_is_available.assert_not_called()
