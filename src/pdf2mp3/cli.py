import argparse
from pathlib import Path
import sys
from . import core
from . import __version__ as package_version

def main():
    """
    Command-Line Interface for pdf2mp3.

    Parses command-line arguments and calls the core conversion function.
    Handles argument validation and provides user feedback.
    """
    parser = argparse.ArgumentParser(
        prog="pdf2mp3", # Explicitly set program name for help messages
        description="Convert a PDF e-book to a single MP3 file using Kokoro TTS.",
        formatter_class=argparse.RawTextHelpFormatter, # Preserves formatting in help text
        epilog="For more information, see README.md or visit the project repository."
    )

    # --- Positional Arguments ---
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Path to the source PDF file."
    )
    parser.add_argument(
        "output_mp3",
        type=Path,
        nargs='?', # Optional
        help="Optional destination file path for the MP3.\n"
             "Defaults to the input PDF's basename with an '.mp3' extension "
             "in the current working directory."
    )

    # --- Core Synthesis Options ---
    synthesis_group = parser.add_argument_group("CORE SYNTHESIS OPTIONS")
    synthesis_group.add_argument(
        "-l", "--lang",
        type=str,
        default="b", # Default to British English code
        help="Target language code (e.g., 'a' for American English, 'b' for British English).\nRefer to README.md for the full list of supported codes.\nDefault: 'b' (British English)"
    )
    synthesis_group.add_argument(
        "-v", "--voice",
        type=str,
        default="bf_emma", # Default from README
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
             "Default: '[.”]\\s*\\n' (splits after periods or quotation marks "
             "followed by optional whitespace and a newline)."
    )

    # --- Audio Encoding Options ---
    audio_group = parser.add_argument_group("AUDIO ENCODING OPTIONS")
    audio_group.add_argument(
        "--bitrate",
        choices=['CONSTANT', 'VARIABLE'],
        type=str.upper, # Convert to uppercase for easier comparison
        default='CONSTANT',
        help="MP3 bitrate control mode.\n  CONSTANT – fixed (CBR) [default]\n  VARIABLE – VBR (average bitrate)"
    )
    audio_group.add_argument(
        "--compression", # Renamed from --compression-float to be more user-friendly
        type=float,
        default=0.5,
        help="Compression level (0.0 to 1.0).\n"
             "Higher values generally result in smaller file sizes (lower bitrate/quality).\n"
             "Lower values aim for better fidelity (higher bitrate/quality).\n"
             "Actual effect depends on the --bitrate mode (CONSTANT or VARIABLE).\n"
             "Default: 0.5"
    )

    # --- Runtime and I/O Options ---
    runtime_group = parser.add_argument_group("RUNTIME AND I/O OPTIONS")
    runtime_group.add_argument(
        "--device",
        type=str,
        default=None, # core.py will auto-detect
        help="Compute device: cpu, cuda, cuda:0, …\nAuto-detected if omitted."
    )
    runtime_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing OUTPUT_MP3."
    )
    runtime_group.add_argument(
        "--resume",
        action="store_true",
        help="Continue a previously interrupted run by reusing\nthe temporary workspace of chunk files."
    )
    runtime_group.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Directory for temporary chunk storage\n(default: “.<output-stem>_chunks” beside OUTPUT)."
    )
    runtime_group.add_argument(
        "--no-progress",
        action="store_false", # Becomes False if flag is present
        dest="show_progress", # Store in 'show_progress'
        default=True, # Default is to show progress
        help="Disable the live progress bar during audio synthesis."
    )

    # --- Miscellaneous Options ---
    misc_group = parser.add_argument_group("MISCELLANEOUS OPTIONS")
    misc_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version}",
        help="Show program's version number and exit."
    )
    # argparse automatically adds --help

    args = parser.parse_args()

    # --- Argument Post-processing and Validation ---

    # Determine output MP3 path if not explicitly provided
    output_mp3_path = args.output_mp3
    if output_mp3_path is None:
        # Default to same name as input PDF but with .mp3 extension, in current dir
        output_mp3_path = Path.cwd() / (args.input_pdf.stem + ".mp3")


    # Validate input PDF file existence
    if not args.input_pdf.is_file():
        parser.error(f"Input PDF file not found: {args.input_pdf}")

    # Validate speed range
    if not (0.5 <= args.speed <= 2.0):
        parser.error(f"Speed must be between 0.5 and 2.0. Got: {args.speed}")

    # Validate compression level range
    if not (0.0 <= args.compression <= 1.0):
        parser.error(f"Compression level must be between 0.0 and 1.0. Got: {args.compression}")

    # Adjust show_progress if quiet mode is enabled
    show_progress_actual = args.show_progress

    # --- Call Core Conversion Logic ---
    try:
        core.convert_pdf_to_mp3(
            pdf_path=args.input_pdf,
            output_mp3_path=output_mp3_path,
            lang=args.lang,
            voice=args.voice,
            speed=args.speed,
            split_pattern=args.split_pattern,
            bitrate_mode=args.bitrate, # Pass the renamed arg
            compression_level=args.compression, # Pass the renamed arg
            device=args.device,
            tmp_dir=args.tmp_dir,
            resume=args.resume,
            show_progress=show_progress_actual, # Use potentially modified value
            overwrite=args.overwrite,
            # Note: The 'quiet' flag itself is not directly passed to core.
            # Instead, its effect (like disabling progress bar) is handled here.
            # Core functions should ideally use a logging setup if granular
            # silence is needed there, or accept a 'verbose'/'quiet' flag.
            # For now, core prints essential info/errors.
        )
    except Exception as e:
        # Catch-all for unexpected errors during core processing
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
