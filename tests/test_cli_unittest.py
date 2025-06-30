import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import io # For capturing stdout/stderr

# Assuming pdf2mp3.cli.main is the entry point
from pdf2mp3.cli import main as cli_main
from pdf2mp3 import __version__ as package_version

class TestCLI(unittest.TestCase):

    def setUp(self):
        """Create a dummy PDF file for testing that needs a file to exist."""
        self.test_dir = Path("tests_temp_cli_data")
        self.test_dir.mkdir(exist_ok=True)
        self.dummy_pdf_path = self.test_dir / "test_book_cli.pdf"
        with open(self.dummy_pdf_path, "w") as f:
            f.write("dummy PDF content")
        self.dummy_output_path = self.test_dir / "output_cli.mp3"

    def tearDown(self):
        """Clean up dummy files and directory."""
        if self.dummy_pdf_path.exists():
            self.dummy_pdf_path.unlink()
        if self.dummy_output_path.exists():
            self.dummy_output_path.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_cli_version(self, mock_stdout):
        """Test the --version flag."""
        # For commands that exit (like --version), we need to catch SystemExit
        with self.assertRaises(SystemExit) as cm:
            with patch.object(sys, 'argv', ['pdf2mp3', '--version']):
                cli_main()
        self.assertEqual(cm.exception.code, 0) # Successful exit
        self.assertIn(f"pdf2mp3 {package_version}", mock_stdout.getvalue())

    @patch("pdf2mp3.cli.core.convert_pdf_to_mp3")
    def test_cli_basic_conversion_call(self, mock_convert):
        """Test basic CLI call, ensuring core.convert_pdf_to_mp3 is called."""
        with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), str(self.dummy_output_path)]):
            cli_main()

        mock_convert.assert_called_once()
        args, kwargs = mock_convert.call_args
        self.assertEqual(kwargs['pdf_path'], self.dummy_pdf_path)
        self.assertEqual(kwargs['output_mp3_path'], self.dummy_output_path)
        self.assertEqual(kwargs['lang'], 'b') # Default lang
        self.assertEqual(kwargs['voice'], 'bf_emma') # Default voice
        self.assertEqual(kwargs['speed'], 0.8) # Default speed

    @patch("pdf2mp3.cli.core.convert_pdf_to_mp3")
    def test_cli_custom_options(self, mock_convert):
        """Test CLI with custom options."""
        custom_lang = "a"
        custom_voice = "test_voice"
        custom_speed = "1.2"
        custom_device = "cpu"

        with patch.object(sys, 'argv', [
            'pdf2mp3', str(self.dummy_pdf_path), str(self.dummy_output_path),
            '--lang', custom_lang,
            '--voice', custom_voice,
            '--speed', custom_speed,
            '--device', custom_device,
            '--overwrite'
        ]):
            cli_main()

        mock_convert.assert_called_once()
        args, kwargs = mock_convert.call_args
        self.assertEqual(kwargs['pdf_path'], self.dummy_pdf_path)
        self.assertEqual(kwargs['output_mp3_path'], self.dummy_output_path)
        self.assertEqual(kwargs['lang'], custom_lang)
        self.assertEqual(kwargs['voice'], custom_voice)
        self.assertEqual(kwargs['speed'], float(custom_speed))
        self.assertEqual(kwargs['device'], custom_device)
        self.assertTrue(kwargs['overwrite'])

    @patch('argparse.ArgumentParser._print_message') # Mocking what parser.error calls
    def test_cli_missing_input_pdf(self, mock_print_message):
        """Test CLI error when input PDF is missing."""
        non_existent_pdf = "non_existent_book.pdf"
        with self.assertRaises(SystemExit) as cm:
            with patch.object(sys, 'argv', ['pdf2mp3', non_existent_pdf]):
                cli_main()
        self.assertNotEqual(cm.exception.code, 0) # Expecting non-zero exit code
        # Check that the error message contains the relevant info
        # The exact message comes from parser.error, which writes to stderr
        # We check if _print_message (internal) was called with a message containing the error
        found_error_message = False
        for call_args in mock_print_message.call_args_list:
            if f"Input PDF file not found: {non_existent_pdf}" in call_args[0][0]:
                found_error_message = True
                break
        self.assertTrue(found_error_message)


    @patch('argparse.ArgumentParser._print_message')
    def test_cli_invalid_speed(self, mock_print_message):
        """Test CLI error for invalid speed."""
        with self.assertRaises(SystemExit) as cm:
            with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), '--speed', '3.0']):
                cli_main()
        self.assertNotEqual(cm.exception.code, 0)
        self.assertTrue(any("Speed must be between 0.5 and 2.0" in call[0][0] for call in mock_print_message.call_args_list))

        mock_print_message.reset_mock()
        with self.assertRaises(SystemExit) as cm:
            with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), '--speed', '0.1']):
                cli_main()
        self.assertNotEqual(cm.exception.code, 0)
        self.assertTrue(any("Speed must be between 0.5 and 2.0" in call[0][0] for call in mock_print_message.call_args_list))


    @patch('argparse.ArgumentParser._print_message')
    def test_cli_invalid_compression(self, mock_print_message):
        """Test CLI error for invalid compression level."""
        with self.assertRaises(SystemExit) as cm:
            with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), '--compression', '1.5']):
                cli_main()
        self.assertNotEqual(cm.exception.code, 0)
        self.assertTrue(any("Compression level must be between 0.0 and 1.0" in call[0][0] for call in mock_print_message.call_args_list))

        mock_print_message.reset_mock()
        with self.assertRaises(SystemExit) as cm:
            with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), '--compression', '-0.5']):
                cli_main()
        self.assertNotEqual(cm.exception.code, 0)
        self.assertTrue(any("Compression level must be between 0.0 and 1.0" in call[0][0] for call in mock_print_message.call_args_list))

    @patch("pdf2mp3.cli.core.convert_pdf_to_mp3")
    def test_cli_default_output_path(self, mock_convert):
        """Test CLI uses default output path when not specified."""
        with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path)]):
            cli_main()

        expected_output_path = Path.cwd() / (self.dummy_pdf_path.stem + ".mp3")
        mock_convert.assert_called_once()
        args, kwargs = mock_convert.call_args
        self.assertEqual(kwargs['output_mp3_path'], expected_output_path)

    @patch("pdf2mp3.cli.core.convert_pdf_to_mp3")
    def test_cli_quiet_mode(self, mock_convert):
        """Test that quiet mode disables progress."""
        with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), '--quiet']):
            cli_main()

        mock_convert.assert_called_once()
        args, kwargs = mock_convert.call_args
        self.assertFalse(kwargs['show_progress'])

    @patch("pdf2mp3.cli.core.convert_pdf_to_mp3")
    def test_cli_no_progress_flag(self, mock_convert):
        """Test that --no-progress flag disables progress."""
        with patch.object(sys, 'argv', ['pdf2mp3', str(self.dummy_pdf_path), '--no-progress']):
            cli_main()

        mock_convert.assert_called_once()
        args, kwargs = mock_convert.call_args
        self.assertFalse(kwargs['show_progress'])

if __name__ == '__main__':
    unittest.main()
