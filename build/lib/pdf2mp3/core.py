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

# Will be refactored from the notebook
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from all pages of a PDF file."""
    text = ""
    with open(pdf_path, "rb") as fh:
        pdf_reader = PyPDF2.PdfReader(fh)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\\n" # Add newline between pages
    return text

def get_device(device_str: str | None = None) -> str:
    """Determines the compute device (CPU or CUDA)."""
    if device_str:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print(f"Warning: CUDA device '{device_str}' requested but not available. Falling back to CPU.")
            return "cpu"
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"

def convert_pdf_to_mp3(
    pdf_path: Path,
    output_mp3_path: Path,
    lang: str = "en-GB",
    voice: str = "bf_emma",
    speed: float = 0.8,
    split_pattern: str = r'[.”]\\s*\\n',
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
    # Note: lang_code 'b' from notebook seems to map to 'en-US' or similar.
    # The README uses 'en-GB'. Let's stick to README for defaults.
    # KPipeline expects specific lang codes, need to check what 'b' was mapping to or if it's a typo.
    # For now, assume `lang` is a valid Kokoro lang code.
    print(f"Initializing Kokoro TTS with language '{lang}' and voice '{voice}'...")
    try:
        pipeline = KPipeline(lang_code=lang, device=actual_device)
    except Exception as e:
        print(f"Error initializing Kokoro pipeline: {e}")
        print("Please ensure the language code and voice are supported by your Kokoro installation.")
        print("You can try listing available languages and voices with --list-langs and --list-voices.")
        return


    # 3. Prepare chunks
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


    # 5. Generate audio
    # The generator from notebook yields (text_chunk, voice_used, lang_used, audio_data)
    # We only need audio_data.
    # The notebook uses tqdm.notebook.tqdm, we should use tqdm.tqdm for CLI.

    # Create a generator for the chunks that still need processing
    def chunk_generator():
        for i, text_chunk in enumerate(chunks[start_chunk_index:], start=start_chunk_index):
            yield text_chunk

    # The pipeline itself can take a list of texts.
    # However, to save intermediate chunks and provide progress, we iterate.

    progress_bar_desc = "Synthesizing"
    current_iteration = start_chunk_index

    # Initialize tqdm progress bar
    pbar = tqdm(total=num_chunks, initial=start_chunk_index, desc=progress_bar_desc, unit="chunk", disable=not show_progress)

    for i, text_chunk in enumerate(chunks[start_chunk_index:], start=start_chunk_index):
        chunk_filename = tmp_dir / f"chunk_{i+1}.npy"
        if resume and chunk_filename.exists() and i < start_chunk_index: # Should have been loaded already
            pbar.update(1)
            continue

        try:
            # Kokoro pipeline processes a single string directly
            # It returns: (text_chunk_processed, voice_used, lang_used, audio_data)
            _, _, _, audio_data = pipeline(
                text_chunk,
                voice=voice,
                speed=speed,
                # split_pattern is applied *before* this step.
                # Kokoro's internal splitting might still occur if a chunk is too long.
            )
            if audio_data is not None and len(audio_data) > 0:
                all_audio_segments.append(audio_data)
                np.save(chunk_filename, audio_data) # Save chunk for resume
            else:
                print(f"Warning: No audio generated for chunk {i+1}. Text: \"{text_chunk[:50]}...\"")
        except Exception as e:
            print(f"Error synthesizing chunk {i+1}: {e}")
            print(f"Text was: \"{text_chunk[:100]}...\"")
            # Optionally, decide if we should stop or continue
        pbar.update(1)

    pbar.close()

    if not all_audio_segments:
        print("No audio was generated or loaded. Cannot create MP3.")
        return

    # 6. Merge and save MP3
    print("Merging audio segments...")
    try:
        merged_audio = np.concatenate(all_audio_segments, axis=0)
    except ValueError as e:
        print(f"Error concatenating audio segments: {e}")
        print("This might happen if some chunks are empty or have inconsistent shapes.")
        return

    print(f"Saving to {output_mp3_path}...")
    try:
        sf.write(
            file=str(output_mp3_path), # soundfile expects string path
            data=merged_audio,
            samplerate=24000,  # Kokoro's fixed sample rate
            format='MP3',
            # soundfile uses 'bitrate' for CBR and VBR quality for VBR.
            # The README implies 'compression_level' controls quality for VBR.
            # Let's assume bitrate_mode='CONSTANT' means CBR and 'VARIABLE' means VBR.
            # soundfile's 'bitrate' parameter for MP3 CBR seems to be in kbps string like '192k'
            # 'compression_level' is not a direct sf.write param for MP3.
            # For VBR, soundfile uses `extra_settings=['-V', str(vbr_quality_0_to_9)]`
            # For CBR, it uses `extra_settings=['-b:a', str(bitrate_in_kbps)]`
            # Let's simplify: use a fixed high-quality VBR or a default CBR.
            # The notebook used: bitrate_mode='CONSTANT', compression_level=.5
            # This does not map directly to soundfile's MP3 options.
            # Let's use a default CBR bitrate for now, e.g., 192kbps.
            # And for VBR, a quality setting.
            # The 'compression_level' from README (0-1) could map to VBR quality (0-9 for lame).
        )
        # The notebook used `bitrate_mode` and `compression_level` which are not standard `sf.write` params.
        # `soundfile` uses `libid3tag` for metadata but doesn't have direct bitrate mode/compression level like that.
        # It passes options to `libsndfile`. For MP3, this often means passing LAME options via `extra_settings`.
        # The notebook example `sf.write` call will likely not work as intended with those params.
        # Let's use a simpler approach for now:
        # sf.write(output_mp3_path, merged_audio, 24000, format='MP3', subtype='MP3')
        # This uses libsndfile's default MP3 settings.
        # To control bitrate for CBR: extra_settings=['-b:a', '192000'] # for 192 kbps
        # For VBR: extra_settings=['-V', '2'] # VBR quality 0-9, 2 is good

        extra_opts = []
        if bitrate_mode.upper() == "VARIABLE":
            # Map compression_level (0-1) to LAME VBR quality (9-0)
            # Higher compression_level (smaller file) means lower VBR quality value in LAME.
            # compression_level 0.0 (best fidelity) -> LAME -V0
            # compression_level 1.0 (smallest file) -> LAME -V9
            vbr_quality = int(round((1.0 - compression_level) * 9))
            extra_opts = ['-V', str(vbr_quality)]
            print(f"Using VBR mode with quality setting: -V {vbr_quality} (derived from compression_level {compression_level})")
        else: # CONSTANT or unspecified
            # Let compression_level (0-1) map to a bitrate range, e.g. 64k to 320k
            # compression_level 0.0 (best fidelity) -> 320kbps
            # compression_level 0.5 (default) -> 192kbps
            # compression_level 1.0 (smallest file) -> 64kbps
            # Simple linear mapping: bitrate = 320 - (compression_level * (320-64))
            cbr_bitrate = int(320 - (compression_level * (320 - 64)))
            extra_opts = ['-b:a', str(cbr_bitrate) + 'k']
            print(f"Using CBR mode with bitrate: {cbr_bitrate}k (derived from compression_level {compression_level})")

        sf.write(
            file=str(output_mp3_path),
            data=merged_audio,
            samplerate=24000, # Kokoro's fixed sample rate
            format='MP3',
            extra_settings=extra_opts
        )
        print(f"Successfully saved MP3 to {output_mp3_path}")

        # Clean up temporary chunk files after successful merge
        if not resume: # Or maybe always clean up if successful?
            print(f"Cleaning up temporary chunk files from {tmp_dir}...")
            for item in tmp_dir.iterdir():
                item.unlink()
            try:
                tmp_dir.rmdir() # Remove the directory itself if empty
            except OSError:
                print(f"Warning: Could not remove temporary directory {tmp_dir}. It might not be empty.")

    except Exception as e:
        print(f"Error saving MP3 file: {e}")

def list_kokoro_languages():
    """Lists available Kokoro languages."""
    # This requires kokoro to have a way to list languages.
    # Assuming KPipeline or another part of kokoro API provides this.
    # Placeholder:
    print("Listing available languages (feature placeholder):")
    # Example of how it might be if kokoro.available_languages() existed:
    # try:
    #     langs = KPipeline.available_languages() # This is hypothetical
    #     for lang_code, lang_name in langs.items():
    #         print(f"  {lang_code}: {lang_name}")
    # except AttributeError:
    #     print("  Kokoro version does not support listing languages directly.")
    # except Exception as e:
    #     print(f"  Could not retrieve languages: {e}")
    print("  en-GB (British English)")
    print("  en-US (American English)")
    print("  es-ES (Castilian Spanish)")
    print("  fr-FR (French)")
    print("  de-DE (German)")
    print("  it-IT (Italian)")
    print("  pt-PT (Portuguese)")
    print("  ja-JP (Japanese)")
    print("  ko-KR (Korean)")
    print("  zh-CN (Mandarin Chinese)")
    print("  (This is a sample list. Actual list depends on your Kokoro setup.)")


def list_kokoro_voices(lang: str | None = None):
    """Lists available Kokoro voices, optionally filtered by language."""
    # This requires kokoro to have a way to list voices.
    # Placeholder:
    print(f"Listing available voices (feature placeholder) for language: {lang if lang else 'all'}")
    # Example of how it might be:
    # try:
    #     voices = KPipeline.available_voices(lang_code=lang) # Hypothetical
    #     for voice_name, voice_details in voices.items():
    #         print(f"  {voice_name}: {voice_details.get('description', 'No description')}")
    # except AttributeError:
    #     print("  Kokoro version does not support listing voices directly.")
    # except Exception as e:
    #     print(f"  Could not retrieve voices: {e}")
    print("  bf_emma (Sample female voice)")
    print("  ms_male_voice (Sample male voice)")
    print("  (This is a sample list. Actual list depends on your Kokoro setup and specified language.)")

if __name__ == '__main__':
    # Example usage (for testing core.py directly)
    # Create a dummy PDF for testing
    from PyPDF2 import PdfWriter

    # Create dummy PDF if it doesn't exist
    dummy_pdf_path = Path("dummy.pdf")
    if not dummy_pdf_path.exists():
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792) # Standard letter size
        # Adding text to a PDF with PyPDF2 is not direct for new PDFs.
        # It's easier to use a tool like reportlab or ensure a text PDF exists.
        # For this test, we'll assume extract_text_from_pdf will be mocked or a real PDF used.
        # Let's manually create a text file and pretend it's PDF content for now.
        with open("dummy_text_for_pdf.txt", "w") as f:
            f.write("This is the first sentence. This is the second sentence.\\n")
            f.write("This is a new paragraph. And another sentence here.")
        print(f"Created dummy_text_for_pdf.txt for testing core.py.")
        print(f"Please manually create a 'dummy.pdf' with some text or use a real PDF for full testing.")

    # Override extract_text_from_pdf for this direct test if dummy.pdf is hard to create with text
    def mock_extract_text_from_pdf(pdf_path: Path) -> str:
        if pdf_path.name == "dummy.pdf" and Path("dummy_text_for_pdf.txt").exists():
            with open("dummy_text_for_pdf.txt", "r") as f:
                return f.read()
        elif pdf_path.exists(): # fallback to actual extraction if not dummy
             with open(pdf_path, "rb") as fh:
                pdf_reader = PyPDF2.PdfReader(fh)
                text = ""
                for page in pdf_reader.pages:
                    text += (page.extract_text() or "") + "\\n"
                return text
        return "This is a test sentence from mock. This is another test sentence."

    original_extractor = extract_text_from_pdf
    extract_text_from_pdf = mock_extract_text_from_pdf

    print("Testing convert_pdf_to_mp3 function...")
    output_file = Path("dummy_output.mp3")
    convert_pdf_to_mp3(
        pdf_path=Path("dummy.pdf"), # Assumes dummy.pdf exists or mock_extract_text works
        output_mp3_path=output_file,
        lang="en-US", # Using a common lang code
        voice="ms_male_voice", # A hypothetical common voice
        speed=1.0,
        overwrite=True,
        show_progress=True
    )
    extract_text_from_pdf = original_extractor # Restore original

    if output_file.exists():
        print(f"Test conversion successful (mocked text, actual TTS). Output: {output_file}")
    else:
        print("Test conversion failed or produced no output.")

    # Test listing functions
    # list_kokoro_languages()
    # list_kokoro_voices("en-US")
