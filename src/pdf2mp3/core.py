# Core functionality for pdf2mp3
import torch
from pypdf import PdfReader
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from tqdm import tqdm
import re
import sys
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text content from a PDF file as a single string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The extracted text as a single string. Returns an empty string if no text can be extracted.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as fh:
            pdf_reader = PdfReader(fh)
            if not pdf_reader.pages:
                return ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n"
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}", file=sys.stderr)
        return ""

    return text


def get_device(device_str: str | None = None) -> str:
    """
    Determine the compute device (CPU or CUDA) for PyTorch.

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
    lang: str = "b",
    voice: str = "bf_emma",
    speed: float = 0.8,
    split_pattern: str = r"[.”]\s*\n",
    bitrate_mode: str = "CONSTANT",
    compression_level: float = 0.5,
    device: str | None = None,
    show_progress: bool = True,
    overwrite: bool = False,
    repo_id: str = "hexgrad/Kokoro-82M",
):
    """
    Convert a PDF file to an MP3 audiobook using Kokoro TTS.

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
    if output_mp3_path.exists() and not overwrite:
        print(f"Error: Output file {output_mp3_path} already exists. Use --overwrite.")
        return

    actual_device = get_device(device)
    print(f"Using device: {actual_device}")

    print(f"Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text could be extracted from the PDF.")
        return
    split_pat = re.compile(split_pattern)
    chunks = [c.strip() for c in split_pat.split(text) if c.strip()]
    num_chunks = len(chunks)
    print(f"Text extracted and split into {num_chunks} chunks.")

    print(
        f"Initializing Kokoro TTS with language code '{lang}', voice '{voice}' on device '{actual_device}'..."
    )
    try:
        pipeline = KPipeline(lang_code=lang, device=actual_device, repo_id=repo_id)
    except Exception as e:
        print(f"Error initializing Kokoro TTS pipeline: {e}")
        print(
            "Please ensure the language code is valid (see README.md) and the voice is supported."
        )
        return

    generator = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=split_pattern,
    )
    all_audio = []
    pbar = tqdm(
        total=num_chunks,
        desc="Synthesizing",
        unit="chunk",
        disable=not show_progress,
    )
    for i, (*_, audio) in enumerate(generator, 1):
        if isinstance(audio, torch.Tensor):
            all_audio.append(audio)
        else:
            print(f"Warning: No valid audio data obtained for chunk {i}.")

        pbar.update(1)
    pbar.close()

    if not all_audio:
        print("No audio segments were generated. Cannot create MP3.")
        return

    print("Merging audio segments...")
    try:
        merged_audio = np.concatenate(all_audio, axis=0)
    except ValueError as e:
        print(f"Error concatenating audio segments: {e}")
        print("This might happen if some chunks are empty or have inconsistent shapes.")
        return

    print(f"Saving merged audio to {output_mp3_path}...")
    try:
        sf.write(
            file=str(output_mp3_path),
            data=merged_audio,
            samplerate=24000,
            format="MP3",
            bitrate_mode=bitrate_mode,
            compression_level=compression_level,
        )
        print(f"Successfully saved MP3 to {output_mp3_path}")
    except Exception as e:
        print(f"Error saving MP3 file: {e}")
