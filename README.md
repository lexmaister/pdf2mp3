# pdf2mp3
Convert a PDF e-book to a single MP3 with Kokoro TTS

`pdf2mp3` is a command-line tool and Python package designed for creating audiobooks from PDF documents. It uses [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M), a robust and versatile text-to-speech engine, to generate natural-sounding audio. The result is a seamless pipeline from book (PDF) to listenable MP3, ideal for audiobook enthusiasts and accessibility applications.

## Features
* PDF to Audiobook: Extracts and reads the text from any standard PDF file.
* Kokoro TTS Engine: Utilizes the Kokoro TTS pipeline for high quality, multilingual audio.
* Custom Voice and Speech Speed: Choose different voices and set custom speaking rates.
* Progress Feedback: Displays a progress bar for chunked synthesis.
* Direct Export: Saves results as an MP3 with proper formatting and bit rate.
* Chunked Processing: Splits the text into manageable chunks for TTS processing, optimizing memory usage.
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

For development (if you plan to modify the source code), you might prefer an editable install:
```bash
pip install -e .
```
This allows your changes in the `src` directory to be reflected immediately without reinstalling.

### Requirements
* Python 3.10+
* pypdf
* kokoro (Kokoro TTS Python package)
* soundfile
* numpy
* tqdm

## Usage

**Basic Conversion:**

Ubuntu/Linux:
```bash
pdf2mp3 /path/to/your/book.pdf /path/to/your/output.mp3 --voice bf_emma --lang b --speed 0.8
```

Windows (PowerShell):
```powershell
pdf2mp3.exe /path/to/your/book.pdf /path/to/your/output.mp3 --voice bf_emma --lang b --speed 0.8
```

If `OUTPUT_MP3` is omitted, the output file will be named after the input PDF (e.g., `book.mp3`) and saved in the current working directory.

**Example with only required argument:**
```bash
pdf2mp3 /path/to/another_book.pdf
# This will create another_book.mp3 in the current directory using default settings.
```

Parameters:
* `INPUT_PDF`: (Required) Path to the source PDF file.
* `OUTPUT_MP3`: (Optional) Path for the generated MP3. Defaults to the input PDF's basename with an `.mp3` extension in the current working directory.
* `--lang <CODE>`: (Optional) Language code for the TTS model. Default: `b` (British English). See "Available Languages" below.
* `--voice <VOICE_NAME>`: (Optional) Voice model selection. Default: `bf_emma`. See "Available Voices" below.
* `--speed <RATIO>`: (Optional) Speech speed ratio (e.g., 0.8 for slower, 1.0 for normal, 1.2 for faster). Default: `0.8`.
* For a full list of options, run `pdf2mp3 --help`.

### Available Languages

The following language codes are supported by the underlying Kokoro TTS engine:

*   `a`: American English 🇺🇸
*   `b`: British English 🇬🇧
*   `e`: Spanish 🇪🇸
*   `f`: French 🇫🇷
*   `h`: Hindi 🇮🇳
*   `i`: Italian 🇮🇹
*   `j`: Japanese 🇯🇵 (Requires `pip install misaki[ja]`)
*   `p`: Brazilian Portuguese 🇧🇷
*   `z`: Mandarin Chinese 🇨🇳 (Requires `pip install misaki[zh]`)

For the most up-to-date list, refer to the [Kokoro TTS documentation](https://github.com/hexgrad/kokoro#advanced-usage).

### Available Voices

A list of available voices can be found on the [Kokoro-82M model card on Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices).
You can also experiment with different voices and languages at the [Kokoro TTS Hugging Face Space](https://hf.co/spaces/hexgrad/Kokoro-TTS).
The default voice is `bf_emma`.

### Full Help Text

For the complete list of all command-line options and their descriptions, run:
```bash
pdf2mp3 --help
```
This will display a detailed help message similar to the following (abbreviated here):
```
USAGE
pdf2mp3 INPUT_PDF [OUTPUT_MP3] [OPTIONS]

FILES (POSITIONAL)
    INPUT_PDF                Path to the source PDF file.
    OUTPUT_MP3               Optional destination file.
                             Defaults to input pdf basename with “.mp3” in current working directory.

CORE SYNTHESIS OPTIONS
  -l, --lang TEXT            Target language code (e.g., 'a' for American English, 'b' for British English).
                             Refer to README.md for the full list of supported codes.
                             Default: 'b' (British English)
  -v, --voice TEXT           Voice preset. Refer to README.md for available voices.
                             Default: bf_emma
  -s, --speed FLOAT          Speaking-rate multiplier (0.5 – 2.0).
                             Default: 0.8
  --split-pattern TEXT       Regular-expression pattern used to split extracted text from the PDF
                             into smaller chunks for TTS processing.
                             Default: '[.”]\\s*\\n'
  --bitrate TEXT             MP3 bitrate control mode.
                               CONSTANT – fixed (CBR) [default]
                               VARIABLE – VBR (average bitrate)
                             Default: CONSTANT
  --compression FLOAT        Compression level (0.0 to 1.0).
                             Higher values generally result in smaller file sizes.
                             Lower values aim for better fidelity.
                             Default: 0.5
  --device TEXT              Compute device: cpu, cuda, cuda:0, …
                             Auto-detected if omitted.
  --overwrite                Replace an existing OUTPUT_MP3.
  --no-progress              Disable the live progress bar.
  --version                  Show program's version number and exit.
  -h, --help                 Show this help message and exit.
```

## Warnings & Notes
*   The application uses the `hexgrad/Kokoro-82M` model by default.
*   Some PyTorch warnings (e.g., related to RNN dropout) may appear during execution; these are generally safe and originate from the underlying TTS library.

## Troubleshooting
* No audio generated: Make sure your PDF contains extractable text.
* Performance: For very large PDFs, ensure you have sufficient RAM or split them into parts.
* Device Support: The tool automatically selects CUDA if available.

## License
[MIT](./LICENSE)

## Authors
[lexmaister](lexmaister@gmail.com)