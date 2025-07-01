import argparse
from pathlib import Path
import sys
# Import core lazily to improve CLI startup speed
# from . import core # This line is commented out
from . import __version__ as package_version

def main():
    """
    Command-Line Interface for pdf2mp3.
    """
    parser = argparse.ArgumentParser(
        prog="pdf2mp3",
        description="Convert a PDF e-book to a single MP3 file using Kokoro TTS.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="For more information, see README.md or visit the project repository."
    )

    # Positional Arguments
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the source PDF file."
    )
    parser.add_argument(
        "output_mp3",
        type=Path,
        nargs='?',
        help="Optional destination file path for the MP3.\n"
             "Defaults to the input PDF's basename with an '.mp3' extension "
             "in the current working directory."
    )

    # Core Synthesis Options
    synthesis_group = parser.add_argument_group("CORE SYNTHESIS OPTIONS")
    synthesis_group.add_argument(
        "-l", "--lang",
        type=str,
        default="b",
        help="Target language code (e.g., 'a' for American English, 'b' for British English).\nRefer to README.md for the full list of supported codes.\nDefault: 'b' (British English)"
    )
    synthesis_group.add_argument(
        "-v", "--voice",
        type=str,
        default="bf_emma",
        help="Voice preset. Refer to README.md for available voices.\nDefault: bf_emma"
    )
    synthesis_group.add_argument(
        "-s", "--speed",
        type=float,
        default=0.8,
        help="Speaking-rate multiplier (0.5 – 2.0).\nDefault: 0.8"
    )
    synthesis_group.add_argument(
        "--split-pattern",
        type=str,
        default=r'[.”]\s*\n',
        help="Regular-expression pattern used to split extracted text from the PDF "
             "into smaller chunks for TTS processing.\n"
             "Default: '[.”]\\s*\\n'"
    )

    # Audio Encoding Options
    audio_group = parser.add_argument_group("AUDIO ENCODING OPTIONS")
    audio_group.add_argument(
        "--bitrate",
        choices=['CONSTANT', 'VARIABLE'],
        type=str.upper,
        default='CONSTANT',
        help="MP3 bitrate control mode.\n  CONSTANT – fixed (CBR) [default]\n  VARIABLE – VBR (average bitrate)"
    )
    audio_group.add_argument(
        "--compression",
        type=float,
        default=0.5,
        help="Compression level (0.0 to 1.0).\n"
             "Higher values generally result in smaller file sizes.\n"
             "Lower values aim for better fidelity.\n"
             "Default: 0.5"
    )

    # Runtime and I/O Options
    runtime_group = parser.add_argument_group("RUNTIME AND I/O OPTIONS")
    runtime_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: cpu, cuda, cuda:0, …\nAuto-detected if omitted."
    )
    runtime_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing OUTPUT_MP3."
    )
    runtime_group.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=True,
        help="Disable the live progress bar."
    )

    # Miscellaneous Options
    misc_group = parser.add_argument_group("MISCELLANEOUS OPTIONS")
    misc_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version}",
        help="Show program's version number and exit."
    )
    # argparse implicitly adds --help

    args = parser.parse_args()

    # Argument Post-processing and Validation
    output_mp3_path = args.output_mp3
    if output_mp3_path is None:
        output_mp3_path = Path.cwd() / (args.input_pdf.stem + ".mp3")

    if not args.input_pdf.is_file():
        parser.error(f"Input PDF file not found: {args.input_pdf}")

    if not (0.5 <= args.speed <= 2.0):
        parser.error(f"Speed must be between 0.5 and 2.0. Got: {args.speed}")

    if not (0.0 <= args.compression <= 1.0):
        parser.error(f"Compression level must be between 0.0 and 1.0. Got: {args.compression}")

    # Call Core Conversion Logic
    try:
        # Import core here to make CLI startup faster when not converting
        from . import core # core is imported here now
        core.convert_pdf_to_mp3(
            pdf_path=args.input_pdf,
            output_mp3_path=output_mp3_path,
            lang=args.lang,
            voice=args.voice,
            speed=args.speed,
            split_pattern=args.split_pattern,
            bitrate_mode=args.bitrate,
            compression_level=args.compression,
            device=args.device,
            # tmp_dir=args.tmp_dir, # Removed
            # resume=args.resume, # Removed
            show_progress=args.show_progress,
            overwrite=args.overwrite,
        )
    except ImportError:
        print("Error: Could not import the 'core' module. Ensure all dependencies, including torch and TTS, are installed correctly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
