import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import numpy as np
import torch

# Assuming src is in PYTHONPATH or using `pip install -e .`
from pdf2mp3 import core

class TestCore(unittest.TestCase):

    def setUp(self):
        # Create a dummy PDF path for tests
        self.dummy_pdf_path = Path("test_dummy.pdf")
        # Create a dummy output path
        self.dummy_output_path = Path("test_output.mp3")
        self.dummy_tmp_dir = Path(".test_output_chunks")

    def tearDown(self):
        # Clean up any files created during tests if necessary
        if self.dummy_output_path.exists():
            self.dummy_output_path.unlink()
        if self.dummy_pdf_path.exists(): # If we actually create it
            self.dummy_pdf_path.unlink()
        if self.dummy_tmp_dir.exists():
            for item in self.dummy_tmp_dir.iterdir():
                item.unlink()
            self.dummy_tmp_dir.rmdir()


    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader_class):
        mock_pdf_instance = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Hello world."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "This is a test."
        mock_pdf_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader_class.return_value = mock_pdf_instance

        # Mock open to simulate file reading for PdfReader
        with patch('builtins.open', mock_open(read_data=b"dummy pdf content")):
            text = core.extract_text_from_pdf(self.dummy_pdf_path)

        self.assertEqual(text, "Hello world.\\nThis is a test.\\n")
        mock_pdf_reader_class.assert_called_once_with(unittest.mock.ANY) # ANY for file handle
        mock_page1.extract_text.assert_called_once()
        mock_page2.extract_text.assert_called_once()

    def test_get_device(self):
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(core.get_device(None), "cuda")
            self.assertEqual(core.get_device("cuda:0"), "cuda:0")
            self.assertEqual(core.get_device("cpu"), "cpu")

        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(core.get_device(None), "cpu")
            self.assertEqual(core.get_device("cpu"), "cpu")
            with patch('builtins.print') as mock_print: # Capture print for warning
                self.assertEqual(core.get_device("cuda:0"), "cpu")
                mock_print.assert_called_with("Warning: CUDA device 'cuda:0' requested but not available. Falling back to CPU.")


    @patch('pdf2mp3.core.KPipeline')
    @patch('pdf2mp3.core.extract_text_from_pdf')
    @patch('soundfile.write')
    @patch('numpy.save') # To mock saving of chunks
    @patch('numpy.load') # To mock loading of chunks for resume
    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open) # General file open mock
    def test_convert_pdf_to_mp3_simple_run(
        self, mock_file_open, mock_path_exists, mock_np_load, mock_np_save, mock_sf_write,
        mock_extract_text, mock_kpipeline_class
    ):
        mock_extract_text.return_value = "This is a sentence. Another sentence."

        mock_pipeline_instance = MagicMock()
        # Simulate Kokoro output: (text_chunk, voice, lang, audio_data)
        # For simplicity, assume one chunk of text becomes one audio segment
        dummy_audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_pipeline_instance.return_value = ("processed_chunk", "voice_used", "lang_used", dummy_audio_data)
        mock_kpipeline_class.return_value = mock_pipeline_instance

        mock_path_exists.return_value = False # Simulate no existing output or chunks

        core.convert_pdf_to_mp3(
            pdf_path=self.dummy_pdf_path,
            output_mp3_path=self.dummy_output_path,
            lang="en-US",
            voice="test_voice",
            speed=1.0,
            overwrite=True, # Important for tests not to fail on existing dummy files
            show_progress=False # Disable progress bar for cleaner test output
        )

        mock_extract_text.assert_called_once_with(self.dummy_pdf_path)
        mock_kpipeline_class.assert_called_once_with(lang_code="en-US", device=core.get_device(None))

        # Check if pipeline was called. Based on split_pattern r'[.”]\\s*\\n'
        # "This is a sentence. Another sentence." -> "This is a sentence", "Another sentence" (approx)
        self.assertEqual(mock_pipeline_instance.call_count, 2)
        mock_pipeline_instance.assert_any_call("This is a sentence", voice="test_voice", speed=1.0)
        mock_pipeline_instance.assert_any_call("Another sentence", voice="test_voice", speed=1.0)

        # Check that numpy.save was called for each chunk
        self.assertEqual(mock_np_save.call_count, 2)
        expected_tmp_dir = Path(f".{self.dummy_output_path.stem}_chunks")
        mock_np_save.assert_any_call(expected_tmp_dir / "chunk_1.npy", dummy_audio_data)
        mock_np_save.assert_any_call(expected_tmp_dir / "chunk_2.npy", dummy_audio_data)

        # Check that soundfile.write was called
        mock_sf_write.assert_called_once()
        args, kwargs = mock_sf_write.call_args
        self.assertEqual(kwargs['file'], str(self.dummy_output_path))
        self.assertEqual(kwargs['samplerate'], 24000)
        self.assertEqual(kwargs['format'], 'MP3')
        # data should be concatenation of the two dummy_audio_data arrays
        expected_merged_audio = np.concatenate([dummy_audio_data, dummy_audio_data])
        np.testing.assert_array_equal(kwargs['data'], expected_merged_audio)


    @patch('pdf2mp3.core.KPipeline')
    @patch('pdf2mp3.core.extract_text_from_pdf')
    @patch('soundfile.write')
    @patch('numpy.save')
    @patch('numpy.load')
    @patch('pathlib.Path.exists') # Mock for Path.exists
    @patch('pathlib.Path.iterdir') # Mock for iterdir to simulate existing chunks
    @patch('builtins.open', new_callable=mock_open)
    def test_convert_pdf_to_mp3_resume_functionality(
        self, mock_open_file, mock_iterdir, mock_path_exists, mock_np_load, mock_np_save,
        mock_sf_write, mock_extract_text, mock_kpipeline_class
    ):
        mock_extract_text.return_value = "Chunk one. Chunk two. Chunk three."

        mock_pipeline_instance = MagicMock()
        audio_data_1 = np.array([0.1, 0.1], dtype=np.float32)
        audio_data_2 = np.array([0.2, 0.2], dtype=np.float32)
        audio_data_3 = np.array([0.3, 0.3], dtype=np.float32)

        # Simulate pipeline calls only for chunks not resumed
        # Chunks are "Chunk one", "Chunk two", "Chunk three"
        # Let's say chunk 1 exists, chunk 2 is synthesized, chunk 3 is synthesized
        mock_pipeline_instance.side_effect = [
            ("processed_chunk2", "v", "l", audio_data_2),
            ("processed_chunk3", "v", "l", audio_data_3),
        ]
        mock_kpipeline_class.return_value = mock_pipeline_instance

        tmp_dir = Path(f".{self.dummy_output_path.stem}_chunks")
        chunk_file_1 = tmp_dir / "chunk_1.npy"
        chunk_file_2 = tmp_dir / "chunk_2.npy" # Will be created
        chunk_file_3 = tmp_dir / "chunk_3.npy" # Will be created

        # Configure Path.exists mock:
        # True for tmp_dir itself, True for chunk_1.npy, False for others initially
        def path_exists_side_effect(path_arg):
            if path_arg == tmp_dir: return True
            if path_arg == chunk_file_1: return True
            if path_arg == self.dummy_output_path : return False # No final output yet
            return False
        mock_path_exists.side_effect = path_exists_side_effect

        # Configure np.load to return audio_data_1 for chunk_1
        mock_np_load.return_value = audio_data_1

        # Mock iterdir to simulate cleanup later (empty list means dir is empty after unlink)
        mock_iterdir.return_value = []


        core.convert_pdf_to_mp3(
            pdf_path=self.dummy_pdf_path,
            output_mp3_path=self.dummy_output_path,
            resume=True,
            overwrite=False, # Important for resume logic
            show_progress=False,
            tmp_dir=tmp_dir
        )

        # Verify np.load was called for the existing chunk
        mock_np_load.assert_called_once_with(chunk_file_1)

        # Verify KPipeline was called only for chunk 2 and 3
        self.assertEqual(mock_pipeline_instance.call_count, 2)
        mock_pipeline_instance.assert_any_call("Chunk two", voice="bf_emma", speed=0.8)
        mock_pipeline_instance.assert_any_call("Chunk three", voice="bf_emma", speed=0.8)

        # Verify np.save was called for newly synthesized chunks (chunk 2 and 3)
        self.assertEqual(mock_np_save.call_count, 2)
        mock_np_save.assert_any_call(chunk_file_2, audio_data_2)
        mock_np_save.assert_any_call(chunk_file_3, audio_data_3)

        # Verify soundfile.write was called with all three segments
        mock_sf_write.assert_called_once()
        args, kwargs = mock_sf_write.call_args
        expected_merged_audio = np.concatenate([audio_data_1, audio_data_2, audio_data_3])
        np.testing.assert_array_equal(kwargs['data'], expected_merged_audio)

    @patch('builtins.print') # To check for "already exists" message
    def test_convert_pdf_to_mp3_output_exists_no_overwrite_no_resume(self, mock_print):
        # Simulate output file already exists
        with patch('pathlib.Path.exists', return_value=True) as mock_path_exists:
            core.convert_pdf_to_mp3(
                pdf_path=self.dummy_pdf_path,
                output_mp3_path=self.dummy_output_path,
                overwrite=False,
                resume=False
            )
            # Check that Path.exists was called on the output_mp3_path
            mock_path_exists.assert_any_call(self.dummy_output_path) # This is fragile if Path.exists is called elsewhere.
                                                                # Better to check the print message.

            mock_print.assert_any_call(f"Error: Output file {self.dummy_output_path} already exists. Use --overwrite or --resume.")

    def test_list_kokoro_languages_placeholder(self):
        with patch('builtins.print') as mock_print:
            core.list_kokoro_languages()
            mock_print.assert_any_call("Listing available languages (feature placeholder):")

    def test_list_kokoro_voices_placeholder(self):
        with patch('builtins.print') as mock_print:
            core.list_kokoro_voices()
            mock_print.assert_any_call("Listing available voices (feature placeholder) for language: all")
            core.list_kokoro_voices(lang="en-US")
            mock_print.assert_any_call("Listing available voices (feature placeholder) for language: en-US")


if __name__ == '__main__':
    unittest.main()
