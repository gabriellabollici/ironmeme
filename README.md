# IRON-Meme

This code corresponds to the project *IRON-Meme: A Benchmark for Structured Understanding of Identities
and Roles in Online Meme Narratives* (2026).

## Prerequisites

- Python >= 3.14

## Getting started

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

This installs the core dependencies plus the default `analysis` group
(numpy, torch, transformers, ...). The `inference` group (vLLM) is a separate
environment for the cluster -- the two can't be installed together, see
`[tool.uv]` in `pyproject.toml`.

To run scripts with a Python other than `.venv`, put the package on the path:
`export PYTHONPATH=src` from the repo root (the VS Code workspace settings in
`.vscode/settings.json` already do this for its terminals).

## Project structure

```
IRONMeme/
├── pyproject.toml, uv.lock           # dependencies: core + analysis/api/dev groups (local), inference (cluster), ocr
├── src/
│   ├── paths.py                      # every repo location (benchmark dataset, sources, results, ...)
│   ├── dataset_construction/         # building the benchmark
│   │   ├── build_benchmark.py        # raw annotations + sources -> benchmark/benchmark_dataset/
│   │   ├── extract_ocr.py            # meme text with EasyOCR (approximates the stored OCR)
│   │   ├── scrapers/                 # Reddit, imgflip and Know Your Meme scrapers (+ config.yaml),
│   │   │                             # kym_full_descriptions / kym_resolve_names (template curation helpers)
│   │   ├── template_identification/  # DINO template embeddings, template matching, dHash clustering
│   │   ├── annotation_formatting/    # raw annotations -> consensus ground truth + agreement type,
│   │   │                             # inter-annotator agreement analysis, blank-viewpoint fills
│   │   └── labels/                   # label inventory and normalisation, aliases.yaml, gt_viewpoint_fills.csv
│   ├── experiments/                  # running and scoring models on the benchmark
│   │   ├── inference/                # 3-step VLM pipeline: run_pipeline (--no-gt), vLLM/API clients,
│   │   │                             # config, models.yaml, prompts/01-03
│   │   ├── evaluation/               # format + score predictions, recompute_all_metrics, diagnostics,
│   │   │                             # random baseline, human evaluation (+ human_evaluators/ annotations)
│   │   ├── template_ablation/        # narrative roles with vs. without the template description (Gemini):
│   │   │                             # ablation.py, compute_metrics.py, prompts/04, 04b
│   │   └── parsing.py                # parsing of raw VLM output
│   ├── benchmark_analysis/           # figures and analyses for the paper (figures in plots/):
│   │                                 # dataset_plots (ground-truth figures), model_comparison_heatmap,
│   │                                 # error_analysis, frequency_vs_performance, benchmark_analysis (stats)
│   ├── slurm/                        # cluster jobs (Snellius) + install_inference_deps.sh, migrate_hf_cache.sh
│   └── tests/                        # regression tests (uv run pytest)
└── benchmark/
    ├── benchmark_dataset/            # THE benchmark (output of build_benchmark):
    │                                 # ironmeme_benchmark.json / .csv, images/ (1110 memes)
    ├── sources/                      # hand-curated / external per-meme data, keyed by annotation id:
    │                                 # templates.csv (template names + descriptions), meme_metadata.csv (OCR, origin)
    ├── source_datasets/              # raw data, local only (not in git): annotation exports, annotated,
    │                                 # scraped and reused memes, template images, KYM sources, embeddings
    └── results/
        ├── raw/                      # model outputs as produced by the runs, never modified
        │   ├── gt_conditioned/       # per-image outputs, one folder per model
        │   └── no_gt/
        ├── metrics/                  # formatted predictions + metrics, rebuilt by recompute_all_metrics
        │   ├── gt_conditioned/       # main results table (all_metrics/, incl. per-identity roles)
        │   ├── no_gt/                # no-ground-truth runs + human evaluators
        │   ├── subset_gt_conditioned/    # VLMs vs. humans on the shared 214-meme subset
        │   └── subset_no_gt/
        ├── template_ablation/        # gemini-3.6-flash/{with_template, no_template}
        └── diagnostics/              # parse failures and discarded labels per run
```

Run modules from the repo root with `uv run python -m <package>.<subpackage>.<module>`,
e.g. `uv run python -m experiments.evaluation.recompute_all_metrics`.

## Usage

### Building the benchmark

```bash
uv run python -m dataset_construction.build_benchmark
```

Writes `benchmark/benchmark_dataset/` (`ironmeme_benchmark.json`, `ironmeme_benchmark.csv`,
`images/`) and nothing else. From the raw annotator exports in
`benchmark/source_datasets/annotation_results/raw_json_annotations/` it derives, in memory, the consensus
ground truth, the agreement type, which memes are included and their order, and copies each
meme's original image under its new name (`<row>.<ext>`). The per-meme data that code cannot derive
is read from `benchmark/sources/` (see below).

### What is not reproducible from code

The template names and descriptions (`benchmark/sources/templates.csv`, `curated=yes`) were curated
by hand. The automatic matching (`template_identification/match_benchmark_templates.py`) only proposes
the nearest template; every name was then checked and corrected where the match or the scraped name
was wrong, and names and descriptions for templates that are not on Know Your Meme were written by
hand. The scripts never overwrite curated rows: the matcher only adds `curated=no` suggestions for
memes without an entry, and `scrapers/kym_full_descriptions.py` only edits those suggestions.

### Scrapers

Both scrapers are configured through a single file,
`src/dataset_construction/scrapers/config.yaml`. Each
scraper has an `enabled` flag so you can run only Reddit, only imgflip, or both:

```yaml
reddit:
  enabled: true
  subreddits:
    - name: TrollXChromosomes
      count: 5
    - name: ClimateMemes
      count: 5

imgflip:
  enabled: true
  template_source: benchmark/source_datasets/memes/template_sources/meme_template_links.parquet
  min_delay: 30
  max_delay: 45

kym:
  enabled: true
  template_source: benchmark/source_datasets/memes/template_sources/kym.json
  min_delay: 35
  max_delay: 45
```

Run all enabled scrapers:

```bash
uv run python -m dataset_construction.scrapers
```

**Reddit scraper** — Each entry specifies a subreddit `name` and the number of
images to download (`count`). Images are saved to
`benchmark/source_datasets/memes/scraped/reddit/<subreddit>/`. The scraper resumes — already
downloaded images are skipped.

**imgflip scraper** — Reads a parquet file (`template_source`, absolute or
relative to the project root) containing a `template_link` column and downloads
each template image into `benchmark/source_datasets/memes/scraped/templates/imgflip/`.
`min_delay`/`max_delay` control the random pause (seconds) between downloads.

**KYM scraper** — Reads a Know Your Meme JSON export (`template_source`, absolute
or relative to the project root), keeps confirmed entries in the "Meme" category
with a `template_image_url`, and downloads each template image into
`benchmark/source_datasets/memes/scraped/templates/kym/`. `min_delay`/`max_delay` control the
random pause (seconds) between downloads.

You can also run a single scraper directly:

```bash
uv run python -m dataset_construction.scrapers.reddit_scraping
uv run python -m dataset_construction.scrapers.imgflip_scraping
uv run python -m dataset_construction.scrapers.kym_scraping
```

### Running the VLMs

Inference runs on the cluster, in its own conda env. The slurm jobs install
the `inference` dependency group (`pip install --group inference`) and put
`src/` on `PYTHONPATH`. From the repo root:

```bash
mkdir -p logs
sbatch src/slurm/run_pipeline.sbatch          # GT-conditioned run
sbatch src/slurm/run_pipeline_no_gt.sbatch    # no-ground-truth run
```

Or directly, against a running vLLM server or a hosted API:

```bash
PYTHONPATH=src python -m experiments.inference.run_pipeline --model Qwen/Qwen3-VL-8B-Instruct
PYTHONPATH=src python -m experiments.inference.run_pipeline --no-gt --limit-images 5
```

Raw outputs go to `benchmark/out/` (or `benchmark/out_no_gt/` with `--no-gt`); move a finished run
into `benchmark/results/raw/gt_conditioned/` (or `.../no_gt/`) to include it in the results.

### Scoring

Rebuild every metrics table under `benchmark/results/` from the raw model outputs:

```bash
uv run python -m experiments.evaluation.recompute_all_metrics
```

Pass `--out-root DIR` to write to a scratch directory instead.

### Tests

```bash
uv run pytest                 # all tests
uv run pytest -m "not slow"   # skip the full metrics recompute (~45 s)
```

They check that the build reproduces `benchmark/benchmark_dataset/` byte for byte, that recomputing
the metrics reproduces every stored result under `benchmark/results/`, that every module the slurm
jobs call exists, and that importing the package writes no files.
