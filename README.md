# pdf2mp3
Convert a PDF e-book to a single MP3 with Kokoro TTS

`pdf2mp3` is a command-line tool and Python package designed for creating audiobooks from PDF documents. It uses Kokoro TTS, a robust and versatile text-to-speech engine, to generate natural-sounding audio. The result is a seamless pipeline from book (PDF) to listenable MP3, ideal for audiobook enthusiasts and accessibility applications.

## Features
* PDF to Audiobook: Extracts and reads the text from any standard PDF file.
* Kokoro TTS Engine: Utilizes the Kokoro TTS pipeline for high quality, multilingual audio.
* Custom Voice and Speech Speed: Choose different voices and set custom speaking rates.
* Progress Feedback: Displays a progress bar for chunked synthesis.
* Direct Export: Saves results as an MP3 with proper formatting and bit rate.
* Chunked Processing: Splits the text into manageable chunks to optimize synthesis and tracking.
* Device-aware: Automatically runs on GPU if available, otherwise uses CPU.

## Installation
Clone the repository
```bash
git clone https://github.com/lexmaister/pdf2mp3
cd pdf2mp3
pip install .
```

### Requirements
* Python 3.10+
* PyPDF2
* kokoro (Kokoro TTS Python package)
* soundfile
* numpy
* tqdm

## Usage
```bash
pdf2mp3  –  Convert a PDF e-book to a single MP3 with Kokoro TTS
-----------------------------------------------------------------

USAGE
    pdf2mp3 [OPTIONS] INPUT_PDF [OUTPUT_MP3]

POSITIONAL ARGUMENTS
    INPUT_PDF                Path to the source PDF file.
    OUTPUT_MP3               Optional destination file.
                             Defaults to INPUT_PDF basename with “.mp3”.

CORE SYNTHESIS
  -l, --lang TEXT            Target language / accent code.  
                             Default: en-GB   (British English)
      --list-langs           List all supported language codes and exit.

  -v, --voice TEXT           Voice preset.  
                             Default: bf_emma
      --list-voices          List available voices and exit.

  -s, --speed FLOAT          Speaking-rate multiplier (0.5 – 2.0).  
                             Default: 0.8

      --split-pattern REGEX  Regular-expression used to split extracted text
                             into synthesis chunks.  
                             Default: '[.”]\\s*\\n'

AUDIO ENCODING
      --bitrate {CONSTANT|VARIABLE}
                             MP3 bitrate control mode.  
                               CONSTANT – fixed (CBR) [default]  
                               VARIABLE – VBR (average bitrate)

      --compression FLOAT    Compression level 0 – 1 (higher = smaller file,
                             lower = better fidelity).  
                             Default: 0.5

RUNTIME & I/O
      --device TEXT          Compute device:  cpu, cuda, cuda:0, …  
                             Auto-detected if omitted.

      --overwrite            Replace an existing OUTPUT_MP3.

      --resume               Continue a previously interrupted run by reusing
                             the temporary workspace of chunk files.

      --tmp-dir PATH         Directory for temporary chunk storage
                             (default: “.<output-stem>_chunks” beside OUTPUT).

      --no-progress          Disable the live progress bar.

MISCELLANEOUS
  -q, --quiet                Suppress non-error console output.
      --version              Show program version and exit.
      --help                 Show this help message and exit.

NOTES
  • Sample rate is fixed at 24 000 Hz by Kokoro models.  
  • Output format is always MP3 in this version.
```

## Warnings & Notes
* By default, the Kokoro TTS repo ID defaults to hexgrad/Kokoro-82M.
* To suppress model selection warnings, specify the repo explicitly.
* Some PyTorch warnings related to RNN dropout and weight normalization may appear; they are safe but should be noted for future updates.

## Troubleshooting
* No audio generated: Make sure your PDF contains extractable text.
* Performance: For very large PDFs, ensure you have sufficient RAM or split them into parts.
* Device Support: The tool automatically selects CUDA if available.

## License
[MIT](./LICENSE)

## Authors
[lexmaister](lexmaister@gmail.com)