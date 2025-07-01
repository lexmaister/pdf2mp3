# Core functionality for pdf2mp3
import torch
import PyPDF2
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from tqdm import tqdm
import re
import sys  # Added for stderr
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path, split_patterns: str) -> tuple[str, ...]:
    """
    Extracts all text content from a given PDF file and splits it into chunks.

    Args:
        pdf_path: Path to the PDF file.
        split_patterns: Regex pattern to split the extracted text.

    Returns:
        A tuple of strings, representing the text chunks.
        Returns an empty tuple if no text can be extracted.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as fh:
            pdf_reader = PyPDF2.PdfReader(fh)
            if not pdf_reader.pages:  # Handle empty or unreadable PDFs early
                return tuple()
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure page_text is not None
                    text += (
                        page_text.strip() + "\\n"
                    )  # Add newline between pages, strip individual page text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}", file=sys.stderr)
        return tuple()
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file {pdf_path}: {e}", file=sys.stderr)
        return tuple()
    except Exception as e:  # Catch other potential errors during PDF processing
        print(
            f"An unexpected error occurred while processing {pdf_path}: {e}",
            file=sys.stderr,
        )
        return tuple()

    if not text.strip():
        return tuple()

    # split_patterns is now a required argument, so we always split.
    compiled_pat = re.compile(split_patterns)
    chunks = tuple(c.strip() for c in compiled_pat.split(text) if c.strip())
    return (
        chunks if chunks else tuple()
    )  # Ensure not to return (None,) or similar if all chunks are empty


def get_device(device_str: str | None = None) -> str:
    """
    Determines the appropriate compute device (CPU or CUDA based) for PyTorch.

    If a specific device string is provided (e.g., "cuda", "cuda:0", "cpu"),
    it attempts to use it. If "cuda" is requested but not available,
    it falls back to "cpu" and prints a warning.
    If no device_str is provided, it defaults to "cuda" if available,
    otherwise "cpu".

    Args:
        device_str: Optional specific device string to request.

    Returns:
        The determined device string (e.g., "cuda", "cpu").
    """
    if device_str:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print(
                f"Warning: CUDA device '{device_str}' requested but not available. Falling back to CPU.",
                file=sys.stderr,
            )
            return "cpu"
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"


def convert_pdf_to_mp3(
    pdf_path: Path,
    output_mp3_path: Path,
    lang: str = "b",  # Default to British English code
    voice: str = "bf_emma",
    speed: float = 0.8,
    split_pattern: str = r"[.”]\s*\n",
    bitrate_mode: str = "CONSTANT",
    compression_level: float = 0.5,
    device: str | None = None,
    # tmp_dir: Path | None = None, # Removed
    # resume: bool = False, # Removed
    show_progress: bool = True,
    overwrite: bool = False,
):
    """
    Converts a PDF file to an MP3 audiobook using Kokoro TTS.

    The process involves:
    1. Extracting text from the PDF.
    2. Initializing the Kokoro TTS pipeline.
    3. Splitting the text into manageable chunks.
    4. Synthesizing audio for each chunk.
    5. Merging audio chunks and saving as an MP3 file.

    Args:
        pdf_path: Path to the source PDF file.
        output_mp3_path: Path to save the generated MP3 file.
        lang: Language code for Kokoro TTS (e.g., 'a', 'b').
        voice: Voice preset for Kokoro TTS.
        speed: Speaking rate multiplier (0.5 – 2.0).
        split_pattern: Regex pattern to split extracted text into chunks.
        bitrate_mode: MP3 bitrate mode ('CONSTANT' or 'VARIABLE').
        compression_level: Compression level (0.0-1.0) influencing bitrate/quality.
        device: Compute device ('cpu', 'cuda', etc.); auto-detected if None.
        show_progress: If True, displays a progress bar during synthesis.
        overwrite: If True, overwrites the output MP3 if it already exists.
    """
    if output_mp3_path.exists() and not overwrite: # Removed 'and not resume'
        print(
            f"Error: Output file {output_mp3_path} already exists. Use --overwrite." # Removed 'or --resume'
        )
        return

    actual_device = get_device(device)
    print(f"Using device: {actual_device}")

    # 1. Extract text
    print(f"Extracting text from {pdf_path}...")
    # Pass split_pattern to extract_text_from_pdf
    chunks = extract_text_from_pdf(pdf_path, split_pattern)

    if not chunks:  # If chunks is empty tuple
        print(
            "No text could be extracted or no text chunks to synthesize after splitting."
        )
        return

    num_chunks = len(chunks)
    print(f"Text extracted and split into {num_chunks} chunks.")

    # 2. Initialize Kokoro TTS pipeline
    print(
        f"Initializing Kokoro TTS with language code '{lang}', voice '{voice}' on device '{actual_device}'..."
    )
    try:
        pipeline = KPipeline(lang_code=lang, device=actual_device)
    except Exception as e:
        print(f"Error initializing Kokoro TTS pipeline: {e}")
        print(
            "Please ensure the language code is valid (see README.md) and the voice is supported."
        )
        return

    # 3. (Text splitting is now done in extract_text_from_pdf)

    # 4. Initialize list to store audio segments
    all_audio_segments = []
    # start_chunk_index is no longer needed as we don't resume.

    # 5. Synthesize audio for each chunk
    progress_bar_desc = "Synthesizing audio chunks"
    # Initialize tqdm with total number of chunks.
    # No 'initial' argument as we always start from the beginning.
    pbar = tqdm(
        total=num_chunks,
        desc=progress_bar_desc,
        unit="chunk",
        disable=not show_progress,
    )

    # Iterate through all chunks.
    for i, text_chunk in enumerate(chunks):
        # chunk_audio_file is no longer needed.
        # Resume logic is removed.

        try:
            # Kokoro pipeline call.
            # The pipeline is called with individual text_chunk.
            # The example `pipeline(text, split_pattern=...)` is for when pipeline handles splitting.
            # Here, text is already pre-split into `chunks`.
            generator_output = pipeline(
                text_chunk,
                voice=voice,
                speed=speed,
                # No split_pattern here as text_chunk is already a small piece of text
            )
            # print(f"DEBUG: pipeline returned for chunk {i+1}: {generator_output}") # Optional debug

            audio_data = None
            # KPipeline returns a generator. We need to iterate through its output.
            # The example suggests it yields (*_, audio_array).
            # We'll assume it yields one item per call if called with a single chunk.
            # Or, if it still returns a generator even for a single chunk, iterate that.

            # Based on the provided example:
            # generator = pipeline(text, ..., split_pattern=split_pat.pattern)
            # for i, (*_, audio) in enumerate(generator, 1): all_audio.append(audio)
            # This implies `pipeline` when given full text + split_pattern returns a generator of multiple audios.
            # However, our current loop calls `pipeline` for *each chunk*.
            # Let's assume `pipeline(one_chunk)` returns the audio directly or a generator that yields one audio.

            # The previous logic for extracting audio_data from generator_output seemed complex.
            # Let's simplify based on the assumption that KPipeline(single_chunk) yields audio data.
            # Typically, a TTS pipeline for a short text segment would produce one audio output.

            # If KPipeline(chunk) returns a generator that yields one audio array:
            if hasattr(generator_output, '__iter__') and not isinstance(generator_output, (str, bytes, dict, np.ndarray)):
                try:
                    # Try to get the first (and presumably only) item from the generator
                    # The example `(*_, audio)` suggests the audio is the last element if a tuple is yielded.
                    for item in generator_output: # Iterate, assuming it might yield multiple things
                        if isinstance(item, np.ndarray):
                            audio_data = item
                            break # Take the first numpy array found
                        elif isinstance(item, (list,tuple)) and len(item) > 0 and isinstance(item[-1], np.ndarray):
                            audio_data = item[-1] # As in (*_, audio_data)
                            break
                    if audio_data is None:
                        # Fallback to the previous complex extraction if simple iteration fails
                        # This is a safety net, ideally KPipeline(chunk) is simpler.
                        first_item = next(iter(generator_output)) # Re-consuming, careful if generator is stateful
                        if isinstance(first_item, np.ndarray):
                            audio_data = first_item
                        elif isinstance(first_item, (list, tuple)) and len(first_item) > 0 and isinstance(first_item[-1], np.ndarray):
                            audio_data = first_item[-1]
                        # else: print(f"Warning: Generator for chunk {i+1} yielded an unexpected item: {first_item}")

                except StopIteration:
                    print(f"Warning: Generator from pipeline was empty for chunk {i+1}.")
                    audio_data = None
                except Exception as e_gen:
                    print(f"Error consuming generator from pipeline for chunk {i+1}: {e_gen}")
                    audio_data = None
            elif isinstance(generator_output, np.ndarray): # If pipeline directly returns an ndarray
                audio_data = generator_output
            else:
                print(f"Warning: Pipeline returned an unexpected type for chunk {i+1}: {type(generator_output)}. Expected a generator or np.ndarray.")
                audio_data = None


            if audio_data is not None and isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                all_audio_segments.append(audio_data)
                # np.save is removed - no saving of temporary chunks.
            else:
                # Simplified warning, removed 'returned_values' as it might be complex.
                print(
                    f'Warning: No valid audio data obtained for chunk {i+1}. Text: "{text_chunk[:50]}..."'
                )
        except Exception as e:
            print(f"Error synthesizing chunk {i+1} ('{text_chunk[:50]}...'): {e}")
        pbar.update(1)

    pbar.close()

    if not all_audio_segments:
        print("No audio segments were generated. Cannot create MP3.")
        return

    # 6. Concatenate audio segments and save to MP3
    print("Merging audio segments...")
    try:
        merged_audio = np.concatenate(all_audio_segments, axis=0)
    except ValueError as e:
        print(f"Error concatenating audio segments: {e}")
        print("This might happen if some chunks are empty or have inconsistent shapes.")
        return

    print(f"Saving merged audio to {output_mp3_path}...")
    try:
        # Determine extra settings for sf.write based on bitrate_mode and compression_level
        # These settings are typically passed to LAME encoder.
        extra_opts = []
        if bitrate_mode.upper() == "VARIABLE":
            # Map compression_level (0.0=best quality, 1.0=smallest file)
            # to LAME VBR quality (0=best, 9=worst).
            # So, compression_level 0.0 -> LAME -V0
            # compression_level 1.0 -> LAME -V9
            vbr_quality = int(round((1.0 - compression_level) * 9))  # Invert and scale
            extra_opts = ["-V", str(vbr_quality)]
            print(
                f"Using Variable Bitrate (VBR) with LAME quality setting -V {vbr_quality} (derived from compression_level {compression_level})."
            )
        else:  # CONSTANT bitrate mode
            # Map compression_level (0.0=best quality/highest bitrate, 1.0=smallest file/lowest bitrate)
            # to a CBR bitrate value, e.g., between 64k and 320k.
            # compression_level 0.0 -> 320kbps
            # compression_level 0.5 -> 192kbps (approx)
            # compression_level 1.0 -> 64kbps
            min_bitrate, max_bitrate = 64, 320
            cbr_bitrate = int(
                max_bitrate - (compression_level * (max_bitrate - min_bitrate))
            )
            extra_opts = ["-b:a", str(cbr_bitrate) + "k"]  # Example: '-b:a 192k'
            print(
                f"Using Constant Bitrate (CBR) at approximately {cbr_bitrate}kbps (derived from compression_level {compression_level})."
            )

        sf.write(
            file=str(output_mp3_path),
            data=merged_audio,
            samplerate=24000,  # Kokoro's fixed sample rate
            format="MP3",
            extra_settings=extra_opts,  # Pass LAME-specific settings
        )
        print(f"Successfully saved MP3 to {output_mp3_path}")

        # 7. Clean up temporary chunk files - This section is removed as temp files are no longer used.

    except Exception as e:
        print(f"Error saving MP3 file: {e}")
