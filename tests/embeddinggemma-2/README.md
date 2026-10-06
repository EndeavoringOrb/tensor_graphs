# EmbeddingGemma 2 reference examples

Run one reference Sentence Transformers inference per modality. The example uses
the local checkpoint at `/mnt/bigdrive/models/google/embeddinggemma-2` by
default, and prints the 768-dimensional embedding shape, norm, and first few
values. It does not require downloading the checkpoint.

The model's saved metadata was produced with Transformers 5.18 development
code. Use a recent Transformers build that supports `EmbeddingGemma2Model`,
plus Sentence Transformers 6.1 or later. Install the model-specific optional
dependencies with:

```bash
uv sync --extra embeddinggemma --extra ml
```

The `ml` extra pins the CUDA 11.8 Torch and TorchVision builds used by this
desktop; PyAV supports the current TorchVision video reader. Generate the
compact, deterministic sample files with FFmpeg and Pillow:

```bash
.venv/bin/python tests/embeddinggemma-2/make_samples.py
```

Then run all four modalities:

```bash
uv run --extra embeddinggemma --extra ml python tests/embeddinggemma-2/run.py
```

Or run one at a time:

```bash
uv run --extra embeddinggemma --extra ml python tests/embeddinggemma-2/run.py text
uv run --extra embeddinggemma --extra ml python tests/embeddinggemma-2/run.py image
uv run --extra embeddinggemma --extra ml python tests/embeddinggemma-2/run.py video
uv run --extra embeddinggemma --extra ml python tests/embeddinggemma-2/run.py audio
```

Pass a different local checkpoint or media file with `--model-path`, `--image`,
`--video`, or `--audio`. Media is passed through Sentence Transformers as
`{"image": path}`, `{"video": path}`, or `{"audio": path}`; text uses the
model's `SearchQuery` prompt. Image, video, and audio inputs receive no text
task prefix, as recommended by the model card.

The video and audio fixtures are intentionally short (3 seconds and 2 seconds)
to keep the reference run small. Transformers currently falls back to
TorchVision for video decoding if TorchCodec is unavailable; that fallback
works here but emits a deprecation warning. Replace the fixtures with real
media paths when investigating more representative outputs.
