# Core functionality for pdf2mp3
import torch
import PyPDF2
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from tqdm import tqdm
import re
import os
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts all text content from a given PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A string containing the extracted text, with newlines between pages.
        Returns an empty string if no text can be extracted.
    """
    text = ""
    with open(pdf_path, "rb") as fh:
        pdf_reader = PyPDF2.PdfReader(fh)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\\n" # Add newline between pages
    return text

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
            print(f"Warning: CUDA device '{device_str}' requested but not available. Falling back to CPU.")
            return "cpu"
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"

def convert_pdf_to_mp3(
    pdf_path: Path,
    output_mp3_path: Path,
    lang: str = "b",  # Default to British English code
    voice: str = "bf_emma",
    speed: float = 0.8,
    split_pattern: str = r'[.”]\s*\n',
    bitrate_mode: str = "CONSTANT",
    compression_level: float = 0.5,
    device: str | None = None,
    tmp_dir: Path | None = None,
    resume: bool = False,
    show_progress: bool = True,
    overwrite: bool = False,
):
    """
    Converts a PDF file to an MP3 audiobook using Kokoro TTS.

    The process involves:
    1. Extracting text from the PDF.
    2. Initializing the Kokoro TTS pipeline.
    3. Splitting the text into manageable chunks.
    4. Synthesizing audio for each chunk, with optional resume capability.
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
        tmp_dir: Directory for temporary chunk storage; defaults to a hidden
                 folder next to the output MP3.
        resume: If True, tries to resume from previously saved chunks.
        show_progress: If True, displays a progress bar during synthesis.
        overwrite: If True, overwrites the output MP3 if it already exists.
    """
    if output_mp3_path.exists() and not overwrite and not resume:
        print(f"Error: Output file {output_mp3_path} already exists. Use --overwrite or --resume.")
        return

    actual_device = get_device(device)
    print(f"Using device: {actual_device}")

    # 1. Extract text
    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text could be extracted from the PDF.")
        return

    # 2. Initialize Kokoro TTS pipeline
    print(f"Initializing Kokoro TTS with language code '{lang}', voice '{voice}' on device '{actual_device}'...")
    try:
        pipeline = KPipeline(lang_code=lang, device=actual_device)
    except Exception as e:
        print(f"Error initializing Kokoro TTS pipeline: {e}")
        print("Please ensure the language code is valid (see README.md) and the voice is supported.")
        return

    # 3. Split text into synthesizable chunks
    compiled_split_pat = re.compile(split_pattern)
    chunks = [c.strip() for c in compiled_split_pat.split(text) if c.strip()]
    num_chunks = len(chunks)

    if num_chunks == 0:
        print("No text chunks to synthesize after splitting.")
        return

    print(f"Text split into {num_chunks} chunks.")

    # 4. Handle temporary directory and resume functionality
    if tmp_dir is None:
        tmp_dir = Path(f".{output_mp3_path.stem}_chunks")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary chunk files will be stored in: {tmp_dir}")

    all_audio_segments = []
    start_chunk_index = 0

    if resume:
        print("Attempting to resume from previously saved chunks...")
        # Load existing audio segments
        for i in range(num_chunks):
            chunk_file = tmp_dir / f"chunk_{i+1}.npy"
            if chunk_file.exists():
                try:
                    audio_segment = np.load(chunk_file)
                    all_audio_segments.append(audio_segment)
                    print(f"Loaded existing chunk {i+1}/{num_chunks}")
                    start_chunk_index = i + 1
                except Exception as e:
                    print(f"Could not load chunk {chunk_file}: {e}. Will re-synthesize.")
                    # Ensure loop continues from this chunk if it failed to load
                    start_chunk_index = i
                    break # Stop loading further chunks if one is corrupt or unreadable
            else:
                # First missing chunk, start synthesis from here
                start_chunk_index = i
                break
        if start_chunk_index > 0:
            print(f"Resuming synthesis from chunk {start_chunk_index + 1}/{num_chunks}")


    # 5. Synthesize audio for each chunk
    # We iterate through chunks to save intermediate results for resumability
    # and to provide progress updates.
    progress_bar_desc = "Synthesizing audio chunks"
    pbar = tqdm(
        total=num_chunks,
        initial=start_chunk_index,
        desc=progress_bar_desc,
        unit="chunk",
        disable=not show_progress
    )

    for i, text_chunk in enumerate(chunks[start_chunk_index:], start=start_chunk_index):
        chunk_audio_file = tmp_dir / f"chunk_{i+1}.npy"

        # If resuming and chunk already processed, skip (it's already loaded)
        if resume and i < start_chunk_index: # This condition ensures we only update pbar for already loaded segments.
            pbar.update(1) # Should ideally be covered by initial pbar setting, but good for safety.
            continue

        try:
            # Kokoro pipeline returns: (processed_text, voice_used, lang_used, audio_data)
            _, _, _, audio_data = pipeline(
                text_chunk,
                voice=voice,
                speed=speed,
            )
            if audio_data is not None and len(audio_data) > 0:
                all_audio_segments.append(audio_data)
                np.save(chunk_audio_file, audio_data) # Save for potential resume
            else:
                print(f"Warning: No audio generated for chunk {i+1}. Text: \"{text_chunk[:50]}...\"")
        except Exception as e:
            print(f"Error synthesizing chunk {i+1} ('{text_chunk[:50]}...'): {e}")
            # Decide whether to stop or continue (currently continues)
        pbar.update(1)

    pbar.close()

    if not all_audio_segments:
        print("No audio segments were generated or loaded. Cannot create MP3.")
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
            vbr_quality = int(round((1.0 - compression_level) * 9)) # Invert and scale
            extra_opts = ['-V', str(vbr_quality)]
            print(f"Using Variable Bitrate (VBR) with LAME quality setting -V {vbr_quality} (derived from compression_level {compression_level}).")
        else:  # CONSTANT bitrate mode
            # Map compression_level (0.0=best quality/highest bitrate, 1.0=smallest file/lowest bitrate)
            # to a CBR bitrate value, e.g., between 64k and 320k.
            # compression_level 0.0 -> 320kbps
            # compression_level 0.5 -> 192kbps (approx)
            # compression_level 1.0 -> 64kbps
            min_bitrate, max_bitrate = 64, 320
            cbr_bitrate = int(max_bitrate - (compression_level * (max_bitrate - min_bitrate)))
            extra_opts = ['-b:a', str(cbr_bitrate) + 'k'] # Example: '-b:a 192k'
            print(f"Using Constant Bitrate (CBR) at approximately {cbr_bitrate}kbps (derived from compression_level {compression_level}).")

        sf.write(
            file=str(output_mp3_path),
            data=merged_audio,
            samplerate=24000,  # Kokoro's fixed sample rate
            format='MP3',
            extra_settings=extra_opts # Pass LAME-specific settings
        )
        print(f"Successfully saved MP3 to {output_mp3_path}")

        # 7. Clean up temporary chunk files
        # Consider always cleaning up if successful, regardless of 'resume' flag,
        # as 'resume' is for input, not for preserving output intermediates indefinitely.
        print(f"Cleaning up temporary chunk files from {tmp_dir}...")
        for item in tmp_dir.iterdir():
            if item.is_file(): # Ensure it's a file before unlinking
                item.unlink()
            try:
                tmp_dir.rmdir() # Remove the directory itself if empty
            except OSError:
                print(f"Warning: Could not remove temporary directory {tmp_dir}. It might not be empty.")

    except Exception as e:
        print(f"Error saving MP3 file: {e}")
