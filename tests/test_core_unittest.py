import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
import numpy as np
# We need to be able to import torch for type hinting and module structure,
# but its functionality will be mocked for CPU-only testing.
import torch

from pdf2mp3.core import extract_text_from_pdf, get_device, convert_pdf_to_mp3

class TestExtractTextFromPDF(unittest.TestCase):
    def setUp(self):
        self.dummy_pdf_path = Path("dummy_core_test.pdf") # Actual file not created, open is mocked

    @patch("PyPDF2.PdfReader")
    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy pdf content")
    def test_extract_text_success(self, mock_file_open, mock_pdf_reader_class):
        """Test successful text extraction."""
        mock_pdf_reader_instance = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Hello world."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "This is a test."
        mock_pdf_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader_class.return_value = mock_pdf_reader_instance

        text = extract_text_from_pdf(self.dummy_pdf_path)

        mock_file_open.assert_called_once_with(self.dummy_pdf_path, "rb")
        self.assertEqual(text, "Hello world.\\nThis is a test.\\n")
        mock_pdf_reader_class.assert_called_once()
        self.assertEqual(mock_page1.extract_text.call_count, 1)
        self.assertEqual(mock_page2.extract_text.call_count, 1)

    @patch("PyPDF2.PdfReader")
    @patch("builtins.open", new_callable=mock_open, read_data=b"dummy pdf content")
    def test_extract_text_empty_or_none(self, mock_file_open, mock_pdf_reader_class):
        """Test extraction when pages return None or empty strings."""
        mock_pdf_reader_instance = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = None
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = ""
        mock_pdf_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader_class.return_value = mock_pdf_reader_instance

        text = extract_text_from_pdf(self.dummy_pdf_path)
        self.assertEqual(text, "\\n\\n") # Newlines are still added


class TestGetDevice(unittest.TestCase):
    @patch("torch.cuda.is_available")
    def test_get_device_cpu_explicitly(self, mock_cuda_available):
        mock_cuda_available.return_value = True # CUDA may be available
        self.assertEqual(get_device("cpu"), "cpu")

    @patch("torch.cuda.is_available")
    def test_get_device_cuda_available_requested(self, mock_cuda_available):
        mock_cuda_available.return_value = True
        self.assertEqual(get_device("cuda"), "cuda")

    @patch("torch.cuda.is_available")
    @patch("builtins.print")
    def test_get_device_cuda_unavailable_requested(self, mock_print, mock_cuda_available):
        mock_cuda_available.return_value = False
        self.assertEqual(get_device("cuda"), "cpu")
        mock_print.assert_called_with("Warning: CUDA device 'cuda' requested but not available. Falling back to CPU.")

    @patch("torch.cuda.is_available")
    def test_get_device_auto_cuda_available(self, mock_cuda_available):
        mock_cuda_available.return_value = True
        self.assertEqual(get_device(), "cuda")

    @patch("torch.cuda.is_available")
    def test_get_device_auto_cpu(self, mock_cuda_available):
        mock_cuda_available.return_value = False
        self.assertEqual(get_device(), "cpu")


@patch("pdf2mp3.core.get_device") # Mock early to control device for all convert tests
@patch("pdf2mp3.core.extract_text_from_pdf")
@patch("pdf2mp3.core.KPipeline")
@patch("numpy.save")
@patch("numpy.load")
@patch("numpy.concatenate")
@patch("soundfile.write")
@patch("pdf2mp3.core.Path.mkdir")
@patch("pdf2mp3.core.Path.exists")
@patch("pdf2mp3.core.Path.iterdir")
@patch("pdf2mp3.core.Path.unlink")
@patch("pdf2mp3.core.Path.rmdir")
@patch("builtins.print") # Suppress print statements
class TestConvertPdfToMp3(unittest.TestCase):

    def setUp(self):
        self.dummy_pdf_path = Path("test_book_core.pdf")
        self.dummy_output_mp3_path = Path("output_core.mp3")
        self.test_tmp_dir = Path(f".{self.dummy_output_mp3_path.stem}_chunks")

        # Default mock behaviors
        self.mock_pipeline_instance = MagicMock()
        self.mock_pipeline_instance.return_value = ("text", "voice", "lang", np.array([0.1, 0.2]))

        self.patch_get_device = None # Will be assigned by class decorator
        self.patch_extract_text = None
        self.patch_kpipeline_class = None
        self.patch_np_save = None
        self.patch_np_load = None
        self.patch_np_concatenate = None
        self.patch_sf_write = None
        self.patch_path_mkdir = None
        self.patch_path_exists = None
        self.patch_path_iterdir = None
        self.patch_path_unlink = None
        self.patch_path_rmdir = None


    def test_normal_flow_cpu(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                             mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                             mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "Sentence one. Sentence two."
        mock_kpipeline_class.return_value = self.mock_pipeline_instance
        mock_path_exists.return_value = False # Output MP3 does not exist
        mock_np_concatenate.return_value = np.array([0.1, 0.2, 0.1, 0.2]) # Mock merged audio

        convert_pdf_to_mp3(
            pdf_path=self.dummy_pdf_path, output_mp3_path=self.dummy_output_mp3_path,
            lang="a", voice="v", speed=1.0, device="cpu", show_progress=False
        )

        mock_extract_text.assert_called_once_with(self.dummy_pdf_path)
        mock_kpipeline_class.assert_called_once_with(lang_code="a", device="cpu")
        self.assertEqual(self.mock_pipeline_instance.call_count, 2)
        self.mock_pipeline_instance.assert_any_call("Sentence one", voice="v", speed=1.0)
        self.mock_pipeline_instance.assert_any_call("Sentence two", voice="v", speed=1.0)
        self.assertEqual(mock_np_save.call_count, 2)
        mock_sf_write.assert_called_once()
        mock_mkdir.assert_called_with(parents=True, exist_ok=True) # For the tmp_dir

    def test_resume_flow(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                         mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                         mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "Chunk one. Chunk two. Chunk three."
        mock_kpipeline_class.return_value = self.mock_pipeline_instance

        chunk1_file = self.test_tmp_dir / "chunk_1.npy"
        chunk2_file = self.test_tmp_dir / "chunk_2.npy"

        # Simulate Path.exists for different paths
        def path_exists_side_effect(path_arg):
            if path_arg == self.dummy_output_mp3_path: return False
            if path_arg == chunk1_file: return True # Chunk 1 exists
            if path_arg == chunk2_file: return False # Chunk 2 does not
            return False
        mock_path_exists.side_effect = path_exists_side_effect
        mock_np_load.return_value = np.array([0.05]) # Loaded audio for chunk 1

        convert_pdf_to_mp3(
            pdf_path=self.dummy_pdf_path, output_mp3_path=self.dummy_output_mp3_path,
            resume=True, show_progress=False, device="cpu"
        )

        mock_np_load.assert_called_once_with(chunk1_file)
        # KPipeline called for chunk 2 and 3 (default lang 'b', voice 'bf_emma', speed 0.8)
        self.assertEqual(self.mock_pipeline_instance.call_count, 2)
        self.mock_pipeline_instance.assert_any_call("Chunk two", voice="bf_emma", speed=0.8)
        self.mock_pipeline_instance.assert_any_call("Chunk three", voice="bf_emma", speed=0.8)
        self.assertEqual(mock_np_save.call_count, 2) # For chunk 2 and 3

    def test_no_text_extracted(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                               mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                               mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "  " # Whitespace only

        convert_pdf_to_mp3(self.dummy_pdf_path, self.dummy_output_mp3_path, show_progress=False)
        mock_print.assert_any_call("No text could be extracted from the PDF.")
        mock_kpipeline_class.assert_not_called() # Pipeline should not initialize

    def test_kpipeline_init_error(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                                  mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                                  mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "Some text."
        mock_kpipeline_class.side_effect = Exception("Kokoro init failed")

        convert_pdf_to_mp3(self.dummy_pdf_path, self.dummy_output_mp3_path, show_progress=False)
        mock_print.assert_any_call("Error initializing Kokoro TTS pipeline: Kokoro init failed")

    def test_output_exists_no_overwrite_no_resume(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                                                   mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                                                   mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_path_exists.return_value = True # Output MP3 exists

        convert_pdf_to_mp3(self.dummy_pdf_path, self.dummy_output_mp3_path, overwrite=False, resume=False)
        mock_print.assert_any_call(f"Error: Output file {self.dummy_output_mp3_path} already exists. Use --overwrite or --resume.")
        mock_extract_text.assert_not_called() # Should exit before extraction

    def test_bitrate_options_vbr(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                                 mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                                 mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "Test."
        mock_kpipeline_class.return_value = self.mock_pipeline_instance
        mock_path_exists.return_value = False
        mock_np_concatenate.return_value = np.array([0.1])

        convert_pdf_to_mp3(
            self.dummy_pdf_path, self.dummy_output_mp3_path,
            bitrate_mode="VARIABLE", compression_level=0.2, device="cpu", show_progress=False
        )
        vbr_quality_expected = int(round((1.0 - 0.2) * 9)) # 7
        mock_sf_write.assert_called_with(
            file=str(self.dummy_output_mp3_path), data=unittest.mock.ANY, #np.array([0.1])),
            samplerate=24000, format='MP3', extra_settings=['-V', str(vbr_quality_expected)]
        )
        mock_print.assert_any_call(f"Using Variable Bitrate (VBR) with LAME quality setting -V {vbr_quality_expected} (derived from compression_level 0.2).")

    def test_bitrate_options_cbr(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                                 mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                                 mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "Test."
        mock_kpipeline_class.return_value = self.mock_pipeline_instance
        mock_path_exists.return_value = False
        mock_np_concatenate.return_value = np.array([0.1])

        convert_pdf_to_mp3(
            self.dummy_pdf_path, self.dummy_output_mp3_path,
            bitrate_mode="CONSTANT", compression_level=0.8, device="cpu", show_progress=False
        )
        min_br, max_br = 64, 320
        cbr_bitrate_expected = int(max_br - (0.8 * (max_br - min_br))) # 115
        mock_sf_write.assert_called_with(
            file=str(self.dummy_output_mp3_path), data=unittest.mock.ANY, #np.array([0.1])),
            samplerate=24000, format='MP3', extra_settings=['-b:a', str(cbr_bitrate_expected) + 'k']
        )
        mock_print.assert_any_call(f"Using Constant Bitrate (CBR) at approximately {cbr_bitrate_expected}kbps (derived from compression_level 0.8).")

    def test_tmp_dir_cleanup(self, mock_print, mock_rmdir, mock_unlink, mock_iterdir, mock_path_exists, mock_mkdir,
                             mock_sf_write, mock_np_concatenate, mock_np_load, mock_np_save,
                             mock_kpipeline_class, mock_extract_text, mock_get_device_func):
        mock_get_device_func.return_value = "cpu"
        mock_extract_text.return_value = "Test."
        mock_kpipeline_class.return_value = self.mock_pipeline_instance
        mock_path_exists.return_value = False
        mock_np_concatenate.return_value = np.array([0.1])

        # Simulate files in tmp_dir
        mock_file_in_tmp_dir = MagicMock(spec=Path)
        mock_file_in_tmp_dir.is_file.return_value = True
        mock_iterdir.return_value = [mock_file_in_tmp_dir]

        convert_pdf_to_mp3(
            self.dummy_pdf_path, self.dummy_output_mp3_path,
            show_progress=False, resume=False # Ensure cleanup is attempted
        )

        mock_iterdir.assert_called_once() # iterdir on the tmp_dir
        mock_unlink.assert_called_once_with(mock_file_in_tmp_dir) # unlink the file
        mock_rmdir.assert_called_once() # rmdir the tmp_dir


if __name__ == '__main__':
    unittest.main()
