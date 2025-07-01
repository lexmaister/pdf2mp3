"""Tests for core pdf2mp3 functionality."""

import pytest
from pathlib import Path
from unittest import mock

# Attempt to import the functions to be tested
# This will allow early feedback if the path or module name is incorrect.
try:
    from pdf2mp3.core import extract_text_from_pdf, get_device, convert_pdf_to_mp3
    import PyPDF2 # For mocking PdfReadError
    import numpy as np # For dummy audio data
    import soundfile as sf # For mocking sf.write
except ImportError as e:
    # If there's an issue with imports, it's critical to know.
    # This could be due to PYTHONPATH issues in the test environment
    # or incorrect module structure.
    # Ensure torch is available or mocked if KPipeline is imported directly
    if "No module named 'torch'" in str(e) and "kokoro" not in str(e):
        # This specific error for torch might be okay if we mock KPipeline for convert_pdf_to_mp3 tests
        print(f"Note: PyTorch not found, KPipeline will need to be mocked for convert_pdf_to_mp3 tests. Error: {e}")
    elif "kokoro" in str(e) or "torch" in str(e):
         print(f"Note: Kokoro or PyTorch not found. KPipeline will need to be mocked. Error: {e}")
    else:
        pytest.fail(f"Failed to import from pdf2mp3.core: {e}. Check PYTHONPATH and module structure.")

# Mock KPipeline globally for test_core.py if torch/kokoro is not installed
# This is a bit broad, but helps if the testing environment can't install them.
# We will explicitly mock it in TestConvertPdfToMp3 anyway.
try:
    from kokoro import KPipeline
except ImportError:
    KPipeline = mock.MagicMock()


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


# --- Fixtures and Tests for convert_pdf_to_mp3 ---

@pytest.fixture
def mock_kpipeline(mocker):
    """Mocks kokoro.KPipeline for testing convert_pdf_to_mp3."""
    mock_pipeline_instance = mocker.MagicMock(spec=KPipeline)
    # Configure the __call__ method (i.e., when the instance is called like a function)
    # It should return a tuple: (processed_text, voice_used, lang_used, audio_data)
    dummy_audio_data = np.array([0.1, 0.2, 0.3] * 1000, dtype=np.float32) # Longer to avoid value errors in concatenate
    mock_pipeline_instance.return_value = ("processed text", "mock_voice", "mock_lang", dummy_audio_data)

    # Mock the class KPipeline to return this instance
    mock_kpipeline_class = mocker.patch('pdf2mp3.core.KPipeline', return_value=mock_pipeline_instance)
    return mock_kpipeline_class, mock_pipeline_instance

@pytest.fixture
def mock_sf_write(mocker):
    """Mocks soundfile.write."""
    return mocker.patch('soundfile.write', spec=sf.write)

@pytest.fixture
def dummy_pdf_path(tmp_path):
    """Creates a simple, valid PDF file with some text for testing."""
    pdf_content_path = tmp_path / "dummy.pdf"
    writer = PyPDF2.PdfWriter()
    writer.add_blank_page(width=612, height=792) # Standard US Letter size
    # Add some text to the page. This is a bit more involved with PyPDF2.
    # For simplicity in this fixture, we'll rely on PyPDF2 to create a readable PDF,
    # and the actual text content can be minimal or just rely on extract_text_from_pdf
    # being tested elsewhere. The main thing is that extract_text_from_pdf returns *something*.
    # To ensure extract_text_from_pdf returns something, let's use a real PDF or mock extract_text_from_pdf.
    # For now, let's assume extract_text_from_pdf works and a blank page is enough
    # for it not to return empty immediately, or we mock extract_text_from_pdf.
    # Let's create a PDF with actual text.

    # Using a more robust way to add text requires reportlab or similar.
    # Instead, we'll create a text file and use a mock for extract_text_from_pdf
    # in tests that need specific text content, or use the hyperion.pdf for integration.
    # This dummy_pdf_path will be for scenarios where the PDF just needs to exist and be basically valid.

    # Let's make it truly simple: a PDF that PyPDF2 can open, and we'll mock extract_text_from_pdf if needed.
    writer.add_blank_page(width=612, height=792)
    with open(pdf_content_path, "wb") as f:
        writer.write(f)
    return pdf_content_path

@pytest.fixture
def output_dir(tmp_path):
    """A temporary directory for output files."""
    d = tmp_path / "output"
    d.mkdir()
    return d


class TestConvertPdfToMp3:
    """Tests for the convert_pdf_to_mp3 function."""

    def test_valid_pdf_normal_processing(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        """TC1: Valid PDF, basic processing flow."""
        output_mp3 = output_dir / "out.mp3"

        # Mock extract_text_from_pdf to return predefined chunks
        mock_extract = mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("chunk1", "chunk2"))

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, lang="b", voice="test_voice", speed=1.0, show_progress=False)

        mock_extract.assert_called_once_with(dummy_pdf_path, r"[.”]\s*\n") # Default split pattern
        mock_kpipeline_class, mock_pipeline_instance = mock_kpipeline
        mock_kpipeline_class.assert_called_once_with(lang_code="b", device="cpu") # Assuming CPU default if torch not fully mocked for cuda

        assert mock_pipeline_instance.call_count == 2 # Called for each chunk
        mock_pipeline_instance.assert_any_call("chunk1", voice="test_voice", speed=1.0)
        mock_pipeline_instance.assert_any_call("chunk2", voice="test_voice", speed=1.0)

        # Temporary directory and chunk file checks are removed.
        # expected_tmp_dir = Path(f".{output_mp3.stem}_chunks")
        # assert expected_tmp_dir.exists()
        # assert expected_tmp_dir.is_dir()
        # assert (expected_tmp_dir / "chunk_1.npy").exists()
        # assert (expected_tmp_dir / "chunk_2.npy").exists()

        mock_sf_write.assert_called_once()
        args, kwargs = mock_sf_write.call_args
        assert kwargs['file'] == str(output_mp3)
        assert kwargs['samplerate'] == 24000
        assert kwargs['format'] == "MP3"
        # Check if audio data is merged (simple check for concatenated length)
        assert len(kwargs['data']) == len(mock_pipeline_instance.return_value[3]) * 2

        # Cleanup of tmp_dir is removed.

        captured = capsys.readouterr()
        assert "Extracting text" in captured.out
        assert "Initializing Kokoro TTS" in captured.out
        assert "Synthesizing audio chunks" in captured.out # This is a tqdm description
        assert "Merging audio segments" in captured.out
        assert f"Successfully saved MP3 to {output_mp3}" in captured.out


    def test_pdf_no_text(self, dummy_pdf_path, output_dir, mock_kpipeline, mocker, capsys):
        """TC2: PDF yields no text."""
        output_mp3 = output_dir / "out_no_text.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=tuple()) # No text extracted

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        assert "No text could be extracted or no text chunks to synthesize" in captured.out
        mock_kpipeline_class, _ = mock_kpipeline
        mock_kpipeline_class.assert_not_called() # Should exit before TTS init
        assert not output_mp3.exists()

    def test_non_existent_pdf(self, tmp_path, output_dir, mock_kpipeline, mocker, capsys):
        """TC3: Non-existent PDF file."""
        non_existent_pdf = tmp_path / "this_does_not_exist.pdf"
        output_mp3 = output_dir / "out_non_existent.mp3"

        # extract_text_from_pdf already handles FileNotFoundError and prints, returns empty.
        # So the behavior inside convert_pdf_to_mp3 will be like test_pdf_no_text
        convert_pdf_to_mp3(non_existent_pdf, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        # Check message from extract_text_from_pdf
        assert f"Error: PDF file not found at {non_existent_pdf}" in captured.err
        # Check message from convert_pdf_to_mp3
        assert "No text could be extracted or no text chunks to synthesize" in captured.out
        mock_kpipeline_class, _ = mock_kpipeline
        mock_kpipeline_class.assert_not_called()
        assert not output_mp3.exists()

    def test_corrupted_pdf(self, empty_pdf_path, output_dir, mock_kpipeline, mocker, capsys):
        """TC4: Corrupted/Invalid PDF file (empty_pdf_path fixture creates a non-PDF)."""
        output_mp3 = output_dir / "out_corrupted.mp3"

        # extract_text_from_pdf handles PdfReadError and prints, returns empty.
        convert_pdf_to_mp3(empty_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        assert f"Error reading PDF file {empty_pdf_path}" in captured.err
        assert "No text could be extracted or no text chunks to synthesize" in captured.out
        mock_kpipeline_class, _ = mock_kpipeline
        mock_kpipeline_class.assert_not_called()
        assert not output_mp3.exists()

    def test_output_exists_no_overwrite(self, dummy_pdf_path, output_dir, capsys): # Renamed test
        """TC6: Output file already exists, overwrite=False.""" # Updated docstring
        output_mp3 = output_dir / "existing_output.mp3"
        output_mp3.touch() # Create dummy existing file

        # Call without resume=False as it's no longer a parameter
        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, overwrite=False, show_progress=False)

        captured = capsys.readouterr()
        # Updated error message check
        assert f"Error: Output file {output_mp3} already exists. Use --overwrite." in captured.out
        # Ensure it doesn't proceed to text extraction
        assert "Extracting text from" not in captured.out

    def test_output_exists_overwrite_true(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        """TC7: Output file already exists, overwrite=True."""
        output_mp3 = output_dir / "overwrite_me.mp3"
        output_mp3.touch()

        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))
        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, overwrite=True, show_progress=False)

        captured = capsys.readouterr()
        assert f"Error: Output file {output_mp3} already exists" not in captured.out
        mock_sf_write.assert_called_once() # Should proceed to write

    # Removed test_tmp_dir_creation_and_cleanup
    # Removed test_resume_no_existing_chunks
    # Removed test_resume_some_existing_chunks
    # Removed test_resume_all_existing_chunks
    # Removed test_resume_corrupted_chunk

    def test_kpipeline_parameters_forwarded(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker):
        """TC14: lang, voice, speed parameters are correctly passed to KPipeline."""
        output_mp3 = output_dir / "params.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("some text",))
        mock_kpipeline_class, mock_pipeline_instance = mock_kpipeline

        # Test with non-default parameters
        custom_lang = "a"
        custom_voice = "test_voice_custom"
        custom_speed = 1.5
        custom_device = "cpu" # Explicitly test CPU

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3,
                           lang=custom_lang, voice=custom_voice, speed=custom_speed,
                           device=custom_device, show_progress=False)

        mock_kpipeline_class.assert_called_once_with(lang_code=custom_lang, device=custom_device)
        mock_pipeline_instance.assert_called_once_with("some text", voice=custom_voice, speed=custom_speed)

        # Removed cleanup for non-existent tmp_dir


    @pytest.mark.parametrize("bitrate_mode, compression_level, expected_vbr_quality, expected_cbr_bitrate_approx_str", [
        ("VARIABLE", 0.0, "0", None),      # Best VBR quality
        ("VARIABLE", 1.0, "9", None),      # Worst VBR quality (smallest file)
        ("VARIABLE", 0.5, "5", None),      # Mid VBR quality ( V = int(round((1-0.5)*9)) = 4 or 5, depends on rounding ) -> (1-0.5)*9 = 4.5 -> 5
        ("CONSTANT", 0.0, None, "320k"),   # Best CBR quality (highest bitrate)
        ("CONSTANT", 1.0, None, "64k"),    # Smallest CBR file (lowest bitrate)
        ("CONSTANT", 0.5, None, "192k"), # Mid CBR quality (approx)
    ])
    def test_mp3_encoding_parameters(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys,
                                     bitrate_mode, compression_level, expected_vbr_quality, expected_cbr_bitrate_approx_str):
        """TC15: Test various bitrate_mode and compression_level settings."""
        output_mp3 = output_dir / f"encoding_{bitrate_mode}_{compression_level}.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text for encoding",))

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3,
                           bitrate_mode=bitrate_mode, compression_level=compression_level,
                           show_progress=False)

        mock_sf_write.assert_called_once()
        _, kwargs = mock_sf_write.call_args
        extra_opts = kwargs.get('extra_settings', [])

        captured = capsys.readouterr()

        if bitrate_mode.upper() == "VARIABLE":
            assert extra_opts == ["-V", str(expected_vbr_quality)]
            assert f"Using Variable Bitrate (VBR) with LAME quality setting -V {expected_vbr_quality}" in captured.out
        else: # CONSTANT
            # Expected CBR is int(320 - (compression_level * (320-64)))
            expected_cbr_val = int(320 - (compression_level * (320 - 64)))
            assert extra_opts == ["-b:a", str(expected_cbr_val) + "k"]
            assert f"Using Constant Bitrate (CBR) at approximately {expected_cbr_val}kbps" in captured.out

        # Removed cleanup for non-existent tmp_dir


    def test_show_progress_false(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker):
        """TC16: tqdm is disabled when show_progress=False."""
        output_mp3 = output_dir / "no_progress.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))
        mock_tqdm = mocker.patch('pdf2mp3.core.tqdm', wraps=tqdm) # Wrap to check call args

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        mock_tqdm.assert_called_once()
        _, tqdm_kwargs = mock_tqdm.call_args
        assert tqdm_kwargs.get('disable') is True

        # Removed cleanup for non-existent tmp_dir

    def test_show_progress_true(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker):
        """Counterpart to TC16: tqdm is enabled when show_progress=True (default)."""
        output_mp3 = output_dir / "progress_shown.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))
        mock_tqdm = mocker.patch('pdf2mp3.core.tqdm', wraps=tqdm)

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=True) # Explicitly True

        mock_tqdm.assert_called_once()
        _, tqdm_kwargs = mock_tqdm.call_args
        assert tqdm_kwargs.get('disable') is False # Default for tqdm is False if not specified, or our func sets it
                                                 # based on show_progress

        # Removed cleanup for non-existent tmp_dir


    def test_device_forwarding_to_kpipeline(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        """TC17, TC18, TC19 (simplified): Check device is passed to KPipeline."""
        output_mp3 = output_dir / "device_test.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))
        mock_kpipeline_class, _ = mock_kpipeline

        # Mock torch.cuda.is_available for predictable behavior from get_device
        # Scenario 1: CUDA available, device=None -> should use cuda
        with mocker.patch('torch.cuda.is_available', return_value=True):
            convert_pdf_to_mp3(dummy_pdf_path, output_mp3, device=None, show_progress=False)
            mock_kpipeline_class.assert_called_with(lang_code='b', device='cuda')
            captured = capsys.readouterr()
            assert "Using device: cuda" in captured.out

        mock_kpipeline_class.reset_mock() # Reset for next call
        # Scenario 2: CUDA not available, device="cuda" -> should use cpu with warning
        with mocker.patch('torch.cuda.is_available', return_value=False):
            convert_pdf_to_mp3(dummy_pdf_path, output_mp3, device="cuda", show_progress=False)
            mock_kpipeline_class.assert_called_with(lang_code='b', device='cpu')
            captured = capsys.readouterr()
            assert "Warning: CUDA device 'cuda' requested but not available." in captured.err
            assert "Using device: cpu" in captured.out

        mock_kpipeline_class.reset_mock()
        # Scenario 3: device="cpu" explicitly
        with mocker.patch('torch.cuda.is_available', return_value=True): # CUDA avail doesn't matter here
            convert_pdf_to_mp3(dummy_pdf_path, output_mp3, device="cpu", show_progress=False)
            mock_kpipeline_class.assert_called_with(lang_code='b', device='cpu')
            captured = capsys.readouterr()
            assert "Using device: cpu" in captured.out

        # Removed cleanup for non-existent tmp_dir

    def test_kpipeline_initialization_failure(self, dummy_pdf_path, output_dir, mock_kpipeline, mocker, capsys):
        """Test behavior when KPipeline initialization fails."""
        output_mp3 = output_dir / "kpipeline_fail.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))

        mock_kpipeline_class, _ = mock_kpipeline
        mock_kpipeline_class.side_effect = Exception("Mocked KPipeline init error")

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        assert "Error initializing Kokoro TTS pipeline: Mocked KPipeline init error" in captured.out
        assert not output_mp3.exists() # Should not create output if TTS fails to init

        # Removed checks and cleanup for non-existent tmp_dir


    def test_synthesis_error_for_one_chunk(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        """Test when one chunk fails synthesis, but others succeed."""
        output_mp3 = output_dir / "chunk_synth_fail.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("good1", "bad_chunk", "good2"))

        mock_kpipeline_class, mock_pipeline_instance = mock_kpipeline
        dummy_audio_data = mock_pipeline_instance.return_value[3]

        def pipeline_side_effect(text_chunk, voice, speed):
            if text_chunk == "bad_chunk":
                raise Exception("Mocked synthesis error")
            return "processed", voice, "lang", dummy_audio_data

        mock_pipeline_instance.side_effect = pipeline_side_effect

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        assert "Error synthesizing chunk 2 ('bad_chunk...'): Mocked synthesis error" in captured.out

        # Chunk file saving checks are removed.
        # tmp_d = Path(f".{output_mp3.stem}_chunks")
        # assert (tmp_d / "chunk_1.npy").exists()
        # assert not (tmp_d / "chunk_2.npy").exists() # Failed one shouldn't be saved
        # assert (tmp_d / "chunk_3.npy").exists()

        # sf.write should still be called with the successfully synthesized chunks
        mock_sf_write.assert_called_once()
        _, write_kwargs = mock_sf_write.call_args
        # Expected audio data from 2 successful chunks
        assert len(write_kwargs['data']) == len(dummy_audio_data) * 2

        # Removed cleanup for non-existent tmp_dir

    def test_no_audio_generated_for_chunk_warning(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        """Test warning when a chunk results in no audio data from TTS."""
        output_mp3 = output_dir / "no_audio_warn.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("good_text", "empty_audio_text"))

        mock_kpipeline_class, mock_pipeline_instance = mock_kpipeline
        dummy_audio_data = mock_pipeline_instance.return_value[3]

        def pipeline_side_effect(text_chunk, voice, speed):
            if text_chunk == "empty_audio_text":
                return "processed", voice, "lang", np.array([]) # Empty audio data
            return "processed", voice, "lang", dummy_audio_data

        mock_pipeline_instance.side_effect = pipeline_side_effect

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        assert 'Warning: No audio generated for chunk 2. Text: "empty_audio_text..."' in captured.out

        # sf.write should be called, but only with data from the first chunk
        mock_sf_write.assert_called_once()
        _, write_kwargs = mock_sf_write.call_args
        assert len(write_kwargs['data']) == len(dummy_audio_data) # Only first chunk's data

        # Removed cleanup for non-existent tmp_dir

    def test_all_chunks_fail_synthesis_or_yield_no_audio(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        """Test scenario where no audio segments are usable for merging."""
        output_mp3 = output_dir / "all_fail_synth.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("fail1", "fail2"))

        mock_kpipeline_class, mock_pipeline_instance = mock_kpipeline
        # Option 1: All chunks raise an error
        # mock_pipeline_instance.side_effect = Exception("Global synth error")
        # Option 2: All chunks return empty audio
        mock_pipeline_instance.return_value = ("processed", "voice", "lang", np.array([]))


        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        # If Option 2 (empty audio for all):
        assert 'Warning: No audio generated for chunk 1' in captured.out
        assert 'Warning: No audio generated for chunk 2' in captured.out
        assert "No audio segments were generated or loaded. Cannot create MP3." in captured.out

        mock_sf_write.assert_not_called() # Should not attempt to write MP3
        assert not output_mp3.exists()

        # Removed cleanup for non-existent tmp_dir

    def test_split_pattern_forwarded_to_extract_text(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker):
        """Test that a custom split_pattern is passed to extract_text_from_pdf."""
        output_mp3 = output_dir / "split_pattern_test.mp3"
        custom_split_pattern = r"\n\n+" # Example: split by double newlines

        mock_extract = mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text based on custom split",))

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, split_pattern=custom_split_pattern, show_progress=False)

        mock_extract.assert_called_once_with(dummy_pdf_path, custom_split_pattern)
        mock_sf_write.assert_called_once() # Ensure it ran through

        # Removed cleanup for non-existent tmp_dir

    # Add a test for invalid output path leading to tmp_dir creation failure if not absolute
    # This test (test_invalid_output_path_tmp_dir_creation_fail) is no longer relevant as tmp_dirs for chunks are not created.
    # def test_invalid_output_path_tmp_dir_creation_fail(self, dummy_pdf_path, tmp_path, mock_kpipeline, mocker, capsys):
    #     """TC8 related: Test when tmp_dir (derived from output_mp3_path) cannot be created."""
        # This output path is problematic because its parent doesn't exist.
        # The default tmp_dir is relative to this, e.g., .nonexistent_dir/output_chunks
        # which Path.mkdir() cannot create if "nonexistent_dir" isn't there.
    #    invalid_output_mp3 = tmp_path / "nonexistent_dir" / "output.mp3"
    #
    #    mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))
    #    mock_kpipeline_class, _ = mock_kpipeline
    #
    #    # We expect an OSError when trying to create the tmp_dir
    #    # For this, we need to let tmp_dir.mkdir() be called.
    #    # The function should ideally catch this and exit or warn.
    #    # Current code does not explicitly catch tmp_dir.mkdir() failure.
    #    # It might proceed and fail later, or raise OSError.
    #    # Let's mock Path.mkdir to simulate the failure more directly for the tmp_dir.
    #
    #    original_path_mkdir = Path.mkdir
    #    def mock_mkdir(self, parents=False, exist_ok=False):
    #        if "chunks" in self.name: # Target the chunk directory
    #             raise OSError("Mocked: Cannot create directory")
    #        return original_path_mkdir(self, parents=parents, exist_ok=exist_ok)
    #
    #    with mocker.patch.object(Path, 'mkdir', side_effect=mock_mkdir, autospec=True):
    #        with pytest.raises(OSError, match="Mocked: Cannot create directory"): # Or check capsys if it's caught
    #            convert_pdf_to_mp3(dummy_pdf_path, invalid_output_mp3, show_progress=False)
    #
    #    # If the function were to catch it and print:
    #    # convert_pdf_to_mp3(dummy_pdf_path, invalid_output_mp3, show_progress=False)
    #    # captured = capsys.readouterr()
    #    # assert "Error creating temporary directory" in captured.err
    #    # mock_kpipeline_class.assert_not_called() # Should fail before TTS

    # Test for sf.write failure
    def test_soundfile_write_failure(self, dummy_pdf_path, output_dir, mock_kpipeline, mock_sf_write, mocker, capsys):
        output_mp3 = output_dir / "sf_write_fail.mp3"
        mocker.patch('pdf2mp3.core.extract_text_from_pdf', return_value=("text",))
        mock_sf_write.side_effect = Exception("Mocked soundfile.write error")

        convert_pdf_to_mp3(dummy_pdf_path, output_mp3, show_progress=False)

        captured = capsys.readouterr()
        assert "Error saving MP3 file: Mocked soundfile.write error" in captured.out

        # Removed checks and cleanup for non-existent tmp_dir
        # tmp_d = Path(f".{output_mp3.stem}_chunks")
        # assert tmp_d.exists() # This would fail as tmp_d is not created
