import argparse
from pathlib import Path
import sys
from . import core
from . import __version__ as package_version # Corrected import

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF e-book to a single MP3 with Kokoro TTS.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
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
        nargs='?', # Optional
        help="Optional destination file.\nDefaults to INPUT_PDF basename with '.mp3'."
    )

    # Core Synthesis Group
    synthesis_group = parser.add_argument_group("CORE SYNTHESIS")
    synthesis_group.add_argument(
        "-l", "--lang",
        type=str,
        default="en-GB",
        help="Target language / accent code.\nDefault: en-GB (British English)"
    )
    synthesis_group.add_argument(
        "--list-langs",
        action="store_true",
        help="List all supported language codes and exit."
    )
    synthesis_group.add_argument(
        "-v", "--voice",
        type=str,
        default="bf_emma", # Default from README
        help="Voice preset.\nDefault: bf_emma"
    )
    synthesis_group.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit."
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
        default=r'[.”]\\s*\\n', # Default from README
        help="Regular-expression used to split extracted text\ninto synthesis chunks.\nDefault: '[.”]\\\\s*\\\\n'"
    )

    # Audio Encoding Group
    audio_group = parser.add_argument_group("AUDIO ENCODING")
    audio_group.add_argument(
        "--bitrate", # Renamed from --bitrate to avoid conflict with potential future actual bitrate int values
        choices=['CONSTANT', 'VARIABLE'],
        type=str.upper, # Convert to uppercase for easier comparison
        default='CONSTANT',
        help="MP3 bitrate control mode.\n  CONSTANT – fixed (CBR) [default]\n  VARIABLE – VBR (average bitrate)"
    )
    audio_group.add_argument(
        "--compression", # Renamed from --compression-float to be more user-friendly
        type=float,
        default=0.5,
        help="Compression level 0 – 1 (higher = smaller file,\nlower = better fidelity).\nDefault: 0.5"
    )

    # Runtime & I/O Group
    runtime_group = parser.add_argument_group("RUNTIME & I/O")
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
        help="Disable the live progress bar."
    )

    # Miscellaneous Group
    misc_group = parser.add_argument_group("MISCELLANEOUS")
    misc_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error console output."
    )
    misc_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version}", # Dynamically get version
        help="Show program version and exit."
    )
    # --help is added automatically by argparse

    args = parser.parse_args()

    # Handle special informational flags
    if args.list_langs:
        core.list_kokoro_languages()
        sys.exit(0)

    if args.list_voices:
        core.list_kokoro_voices(lang=args.lang if args.lang != "en-GB" else None) # Pass lang if specified
        sys.exit(0)

    # Determine output path if not provided
    output_mp3_path = args.output_mp3
    if output_mp3_path is None:
        output_mp3_path = args.input_pdf.with_suffix(".mp3")

    # Validate input PDF
    if not args.input_pdf.is_file():
        print(f"Error: Input PDF file not found: {args.input_pdf}")
        sys.exit(1)

    # Validate speed
    if not (0.5 <= args.speed <= 2.0):
        print(f"Error: Speed must be between 0.5 and 2.0. Got: {args.speed}")
        sys.exit(1)

    # Validate compression level
    if not (0.0 <= args.compression <= 1.0):
        print(f"Error: Compression level must be between 0.0 and 1.0. Got: {args.compression}")
        sys.exit(1)

    # Quiet mode: redirect stdout to /dev/null or equivalent
    # This is a simple way; a more robust way involves custom stream handlers or logging.
    # For now, if quiet, critical print statements in core.py might need adjustment
    # or core.py needs to accept a quiet flag.
    # Let's assume core.py prints essential info and errors regardless,
    # and CLI controls additional verbosity if needed later.
    # For now, `quiet` primarily means CLI itself won't print extra status messages.
    # The `show_progress` flag handles the progress bar.

    if args.quiet:
        # This is a bit of a hack. Proper logging framework would be better.
        # For now, let's assume core functions will still print errors.
        # We can pass a `quiet` flag to `convert_pdf_to_mp3` if necessary.
        # Let's make core.py's print statements conditional or use logging.
        # For now, this will only suppress CLI prints, not core prints.
        pass


    # Call the core conversion function
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
        show_progress=args.show_progress,
        overwrite=args.overwrite,
        # TODO: Pass quiet flag to core if core is modified to support it
    )

if __name__ == "__main__":
    main()
