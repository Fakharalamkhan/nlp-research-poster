"""Builds the Kaggle notebook for cgsc-run using an isolated venv."""

import os
import json
import gzip
import base64
import nbformat as nbf


def make_notebook(output_path: str = "kaggle/cgsc_run.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.10"}

    # Load prompts
    prompts_dir = "cgsc/prompts"
    prompt_files = {
        "a0_gsm8k.txt": open(os.path.join(prompts_dir, "a0_gsm8k.txt"), "r", encoding="utf-8").read(),
        "a0_hotpotqa.txt": open(os.path.join(prompts_dir, "a0_hotpotqa.txt"), "r", encoding="utf-8").read(),
        "verb.txt": open(os.path.join(prompts_dir, "verb.txt"), "r", encoding="utf-8").read(),
        "verif_gsm8k.txt": open(os.path.join(prompts_dir, "verif_gsm8k.txt"), "r", encoding="utf-8").read(),
        "verif_hotpotqa.txt": open(os.path.join(prompts_dir, "verif_hotpotqa.txt"), "r", encoding="utf-8").read(),
        "revision.txt": open(os.path.join(prompts_dir, "revision.txt"), "r", encoding="utf-8").read(),
        "ioe.txt": open(os.path.join(prompts_dir, "ioe.txt"), "r", encoding="utf-8").read(),
    }

    # Load splits
    with open("cgsc/data/splits.json", "r", encoding="utf-8") as f:
        splits_json_str = f.read()

    # Load and compress local gsm8k dataset (130 KB b64)
    with open("cgsc/data/gsm8k.jsonl", "rb") as f:
        gsm8k_b64 = base64.b64encode(gzip.compress(f.read(), 9)).decode("ascii")

    # Load scoring, sanity, run code
    with open("cgsc/scoring.py", "r", encoding="utf-8") as f:
        scoring_code = f.read()
    with open("cgsc/sanity.py", "r", encoding="utf-8") as f:
        sanity_code = f.read()
    with open("cgsc/run.py", "r", encoding="utf-8") as f:
        run_code = f.read()

    # Cell 0: Parameters
    c0 = """# PARAMETERS CELL
SETTINGS = [
    "qwen/gsm8k",
    "qwen/hotpotqa",
]

DRY_RUN_COUNT = 20
FULL_RUN_COUNT = 500
OUTPUT_DIR = "/kaggle/working"
"""

    # Cell 1: Install uv and create isolated venv with vllm
    c1 = """import subprocess, sys, os

print("=" * 60)
print("1. INSTALLING UV & CREATING ISOLATED VIRTUAL ENVIRONMENT")
print("=" * 60)

# Set HF_HOME to /tmp/hf so model weights do not clutter /kaggle/working
os.environ["HF_HOME"] = "/tmp/hf"
os.makedirs("/tmp/hf", exist_ok=True)

# Install uv into system python (fast, standalone, doesn't touch system libraries)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])

# Create clean virtual environment with Python 3.11 in /tmp/venv
subprocess.check_call(["uv", "venv", "/tmp/venv", "--python", "3.11"])

# Install vllm and dependencies inside /tmp/venv
print("\\nInstalling vllm and dependencies inside /tmp/venv...")
res = subprocess.run([
    "uv", "pip", "install", "--python", "/tmp/venv/bin/python",
    "vllm==0.6.6.post1", "setuptools", "transformers==4.46.1", "pyarrow<19.0.0,>=15.0.0", "datasets>=3.0.0", "pandas", "scipy", "scikit-learn", "pyyaml"
])
if res.returncode != 0:
    print("Fallback: Installing vllm==0.7.3 into /tmp/venv...")
    subprocess.check_call([
        "uv", "pip", "install", "--python", "/tmp/venv/bin/python",
        "vllm==0.7.3", "setuptools", "transformers==4.46.1", "pyarrow<19.0.0,>=15.0.0", "datasets>=3.0.0", "pandas", "scipy", "scikit-learn", "pyyaml"
    ])

print("\\nIsolated virtual environment ready at /tmp/venv.")
"""

    # Cell 2: Environment verification
    c2 = """import subprocess, sys

print("=" * 60)
print("2. VIRTUAL ENVIRONMENT & HARDWARE VERIFICATION")
print("=" * 60)
subprocess.run([
    "/tmp/venv/bin/python", "-c",
    "import numpy, torch, vllm, transformers; print(f'PyTorch: {torch.__version__}, NumPy: {numpy.__version__}, vLLM: {vllm.__version__}, Transformers: {transformers.__version__}')"
], check=True)

print("\\nNVIDIA-SMI OUTPUT:")
subprocess.run(["nvidia-smi"], check=True)
print("=" * 60)
sys.stdout.flush()
"""

    # Cell 3: Setup CGSC package files and datasets on Kaggle
    c3 = f"""# 3. PREPARE CGSC PACKAGE FILES & DATASETS
import os, sys, shutil, gzip, base64

os.makedirs("cgsc/data", exist_ok=True)
os.makedirs("cgsc/prompts", exist_ok=True)
os.makedirs("results/raw", exist_ok=True)

with open("cgsc/__init__.py", "w", encoding="utf-8") as f:
    f.write('\"\"\"CGSC Package\"\"\"\\n')

with open("cgsc/scoring.py", "w", encoding="utf-8") as f:
    f.write({repr(scoring_code)})

with open("cgsc/sanity.py", "w", encoding="utf-8") as f:
    f.write({repr(sanity_code)})

with open("cgsc/data/splits.json", "w", encoding="utf-8") as f:
    f.write({repr(splits_json_str)})

# Write pre-split local gsm8k dataset
with open("cgsc/data/gsm8k.jsonl", "wb") as f:
    f.write(gzip.decompress(base64.b64decode({repr(gsm8k_b64)}.encode("ascii"))))

with open("cgsc/run.py", "w", encoding="utf-8") as f:
    f.write({repr(run_code)})

prompt_data = {repr(prompt_files)}
for fname, content in prompt_data.items():
    with open(os.path.join("cgsc/prompts", fname), "w", encoding="utf-8") as f:
        f.write(content)

# Mount original raw Qwen files from attached Kaggle dataset fakharalam1/cgsc-raw
raw_src_dir = "/kaggle/input/cgsc-raw"
for f in ["qwen_gsm8k.jsonl", "qwen_hotpotqa.jsonl"]:
    src_f = os.path.join(raw_src_dir, f)
    dst_f = os.path.join("results/raw", f)
    if os.path.exists(src_f):
        print(f"Copying {{f}} from {{raw_src_dir}} to results/raw/...")
        shutil.copy(src_f, dst_f)
    elif os.path.exists(f):
        shutil.copy(f, dst_f)
    else:
        print(f"Warning: {{src_f}} not found; checking working directory...")

print(f"results/raw contents: {{os.listdir('results/raw')}}")
print("cgsc package files & datasets successfully initialized on disk.")
"""

    # Cell 4: Orchestrator Loop
    c4 = """import os, sys, subprocess

print("=" * 70)
print("4. EXECUTING CGSC PIPELINE VIA ISOLATED SUBPROCESS")
print("=" * 70)

# Configure environment for subprocess
sub_env = os.environ.copy()
sub_env["HF_HOME"] = "/tmp/hf"
sub_env["VLLM_USE_V1"] = "0"

# Securely retrieve HF_TOKEN from kaggle_secrets without printing
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_tok = user_secrets.get_secret("HF_TOKEN")
    if hf_tok:
        sub_env["HF_TOKEN"] = hf_tok
        sub_env["HUGGING_FACE_HUB_TOKEN"] = hf_tok
        print("HF_TOKEN successfully loaded from Kaggle secrets.")
    else:
        print("HF_TOKEN secret not found or empty.")
except Exception as e:
    print(f"Notice: UserSecretsClient returned: {e}")

errors_file = os.path.join(OUTPUT_DIR, "errors.txt")

for setting_str in SETTINGS:
    if ":resample" in setting_str:
        base = setting_str.replace(":resample", "")
        model_key, dataset_key = base.split("/")
        is_resample = True
    else:
        model_key, dataset_key = setting_str.split("/")
        is_resample = False

    setting_tag = f"{model_key}_{dataset_key}"
    mode_desc = "RESAMPLE SC ONLY" if is_resample else "FULL PIPELINE"
    print(f"\\n{'#' * 70}")
    print(f"# STARTING SETTING: {setting_tag} ({mode_desc})")
    print(f"{'#' * 70}\\n")
    sys.stdout.flush()

    cmd = [
        "/tmp/venv/bin/python", "-m", "cgsc.run",
        "--model", model_key,
        "--dataset", dataset_key,
        "--dry_count", str(DRY_RUN_COUNT),
        "--full_count", str(FULL_RUN_COUNT),
        "--output_dir", OUTPUT_DIR,
        "--raw_dir", "results/raw"
    ]
    if is_resample:
        cmd.append("--resample_sc_only")

    res = subprocess.run(cmd, env=sub_env)
    if res.returncode != 0:
        err_msg = f"Setting {setting_tag} failed with exit code {res.returncode}"
        print(f"\\nERROR: {err_msg}")
        with open(errors_file, "a", encoding="utf-8") as ef:
            ef.write(f"[{setting_tag}] {err_msg}\\n")
        print(f"Skipping {setting_tag} and continuing to next setting.")
    else:
        print(f"\\nSUCCESS: Setting {setting_tag} completed successfully!")

    sys.stdout.flush()

print("\\n" + "=" * 70)
print("ALL REQUESTED SETTINGS PROCESSED!")
print("=" * 70)
"""

    nb.cells = [
        nbf.v4.new_code_cell(c0),
        nbf.v4.new_code_cell(c1),
        nbf.v4.new_code_cell(c2),
        nbf.v4.new_code_cell(c3),
        nbf.v4.new_code_cell(c4),
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {output_path} (size: {os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    make_notebook()
