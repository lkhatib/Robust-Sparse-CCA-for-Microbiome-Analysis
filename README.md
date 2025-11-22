RoSCA Pipeline

A reproducible Snakemake workflow for running the Robust Sparse Canonical Analysis (RoSCA) model on microbiome and dietary datasets.

This repository provides an end-to-end, reproducible workflow for running RoSCA on the THDMI dataset (or any microbiome × metadata dataset).
It includes:

A Snakemake workflow

A conda environment (envs/rosca.yaml)

All analysis scripts (src/rosca_pipeline/)

A minimal test dataset (tests/thdmi/)

Optional hyperparameter tuning + plotting modules

The workflow performs:
preprocessing → model fitting → optional tuning → output → optional plotting

🔧 Repository Structure
rosca-pipeline/
├── Snakefile
├── README.md
├── .gitignore
├── envs/
│   └── rosca.yaml
├── config/
│   └── thdmi.yaml
├── src/
│   └── rosca_pipeline/
│       ├── rosca.py
│       ├── preprocessing.py
│       ├── auto_tuning.py
│       └── plotting.py
├── notebooks/
│   └── plotting.ipynb
├── scripts/
│   └── run_RoSCA_example.sh
├── tests/
│   └── thdmi/
│       ├── data/
│       │   ├── thdmi_feature-table_filtered_samples_features.biom
│       │   └── nutrients_data_no_cal.csv
│       ├── expected/
│       │   ├── THDMI.summary.json
│       │   └── THDMI.fit_warning.txt
│       └── output/   # Snakemake writes here
└── results/          # Large real outputs (gitignored)

🚀 Quickstart
Install Snakemake
conda install -c conda-forge snakemake

Run the full workflow

From the repository root:

snakemake --use-conda --cores 4


Snakemake will:

Create the environment from envs/rosca.yaml

Load the biom + metadata files

Run RoSCA

Produce:

THDMI.summary.json

THDMI.fit_warning.txt

⚙️ Configuration

All dataset-specific paths live in:

config/thdmi.yaml


Example:

dataset: "thdmi"
X: "tests/thdmi/data/thdmi_feature-table_filtered_samples_features.biom"
Y: "tests/thdmi/data/nutrients_data_no_cal.csv"
outdir: "tests/thdmi/output"


Modify these paths to run the pipeline on your own dataset.

📦 Pipeline Overview
Rule: run_rosca

Runs the RoSCA model.

Produces:

THDMI.summary.json

THDMI.fit_warning.txt

Command structure:

python -m rosca_pipeline.rosca \
    --X <input_features> \
    --Y <input_metadata> \
    --out-directory <output_dir>

Optional Rules

preprocess — sample/feature filtering

tune_rosca — automated hyperparameter tuning (Optuna)

plot_rosca — executes plotting notebook or script

🧪 Testing with Included Dataset

A miniature THDMI dataset is included:

tests/thdmi/data/


To dry-run:

snakemake -n


To run only on test data:

snakemake --cores 2


Expected outputs for regression testing exist in:

tests/thdmi/expected/

🧰 Development
Add new rules

Edit Snakefile:

rule my_step:
    input:
    output:
    conda:
    shell:

Run a specific rule
snakemake run_rosca --use-conda

Remove outputs
snakemake --delete-all-output

📤 Version Control

Typical Git setup:

git init
git add .
git commit -m "Initial Snakemake RoSCA pipeline"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main

📚 Citation

If you use this pipeline or RoSCA in your work, please cite the original RoSCA method and this repository.

📧 Contact

For questions or contributions, please open an issue or submit a pull request.
