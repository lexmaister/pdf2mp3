# pdf2mp3
Convert a PDF e-book to a single MP3 with Kokoro TTS

`pdf2mp3` is a command-line tool and Python package designed for creating audiobooks from PDF documents. It uses [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M), a robust and versatile text-to-speech engine, to generate natural-sounding audio. The result is a seamless pipeline from book (PDF) to listenable MP3, ideal for audiobook enthusiasts and accessibility applications.

## Features
* PDF to Audiobook: Extracts and reads the text from any standard PDF file.
* Kokoro TTS Engine: Utilizes the Kokoro TTS pipeline for high quality, multilingual audio.
* Custom Voice and Speech Speed: Choose different voices and set custom speaking rates.
* Progress Feedback: Displays a progress bar for chunked synthesis.
* Direct Export: Saves results as an MP3 with proper formatting and bit rate.
* Chunked Processing: Splits the text into manageable chunks to optimize synthesis and tracking.
* Device-aware: Automatically runs on GPU if available, otherwise uses CPU.

## Installation
Clone the repository, setup virtual environment and run it.

Ubuntu:
```bash
git clone https://github.com/lexmaister/pdf2mp3
cd pdf2mp3
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Windows (PowerShell):
```powershell
git clone https://github.com/lexmaister/pdf2mp3
cd pdf2mp3
python -m venv .venv
.\.venv\Scripts\activate
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

Ubuntu:
```bash
pdf2mp3 --pdf /path/to/book.pdf --output /path/to/book.mp3 --voice bf_emma --lang b --speed 0.8
```

Windows:
```powershell
pdf2mp3.exe --pdf /path/to/book.pdf --output /path/to/book.mp3 --voice bf_emma --lang b --speed 0.8
```

Parameters:
* `--pdf`: Path to your PDF book.
* `--output`: Path for the generated MP3.
* `--voice`: (Optional) Voice model selection. Default: bf_emma.
* `--lang`: (Optional) Language code for TTS model.
* `--speed`: (Optional) Speech speed ratio (1.0 is normal, 0.8 is slower).

### Available languages
```
# 🇺🇸 'a' => American English 
# 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
```

More supported languages - see in [Kokoro documentation](https://github.com/hexgrad/kokoro#advanced-usage)


### Available voices

This list can found be [here](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices). Also you can try different languages [here](https://hf.co/spaces/hexgrad/Kokoro-TTS).

### Full help
```bash
pdf2mp3  –  Convert a PDF e-book to a single MP3 with Kokoro TTS
-----------------------------------------------------------------

USAGE
pdf2mp3 INPUT_PDF [OUTPUT_MP3] [OPTIONS]

FILES (POSITIONAL)
    INPUT_PDF                Path to the source PDF file.
    OUTPUT_MP3               Optional destination file.
                             Defaults to input pdf basename with “.mp3” in current working directory.

CORE SYNTHESIS
  -l, --lang TEXT            Target language / accent code.  
                             Default: en-GB   (British English)

  -v, --voice TEXT           Voice preset.  
                             Default: bf_emma

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
  -h, --help                 Show this help message and exit.

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