# pdf2mp3
Convert a PDF e-book to a single MP3 with Kokoro TTS

## Options
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
