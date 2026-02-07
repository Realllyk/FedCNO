# Repository Guidelines

## Project Structure & Module Organization
- `fed_main/` contains the main experiment entry points (e.g., `Fed_LGV.py`, `Fed_Avg.py`).
- `models/`, `trainers/`, and `utils/` hold core model definitions, training loops, and shared utilities.
- `data/` and `data_processing/` store datasets and preprocessing code.
- `bash_scripts/` includes runnable experiment shells that wrap common command lines.
- `runs/`, `graduate_result/`, `merge_result/`, and `4_client_result/` collect outputs and metrics.

## Build, Test, and Development Commands
- Create an environment from conda-style pins in `requirements.txt` (Python 3.7, PyTorch 1.12 CUDA 11.3).
- Run a main experiment directly:
  `python fed_main/Fed_LGV.py --vul reentrancy --noise_type non_noise --noise_rate 0.3 --epoch 30 --warm_up_epoch 25 --batch 8 --random_noise --global_weight 0.75 --num_neigh 5 --lab_name Fed_LGV --model_type CBGRU --diff --consistency_score`
- Use the bash wrappers for repeatable sweeps:
  `bash bash_scripts/run_Fed_LGV.sh <device> <num_neigh>`

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and `snake_case` for functions/variables.
- Module names and scripts are `PascalCase` or `snake_case` depending on experiment (`Fed_LGV.py`, `Fed_Avg.py`).
- No formatter or linter is configured; keep edits consistent with nearby style.

## Testing Guidelines
- No dedicated test framework (pytest/unittest) is present.
- Evaluation is performed through training runs and `global_test.py`, with metrics written to result folders (e.g., `graduate_result/<lab>/<model>/<noise_type>/<noise_rate>`).
- For new features, add a minimal run command and verify the produced JSON metrics.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative summaries (e.g., `fix clc`, `fixed CLC`, `优化CLC的CPU占用`).
- Keep messages concise and focused on the change.
- For PRs, include a brief description, key commands run, and attach sample result files or screenshots of metrics if behavior changes.

## Security & Configuration Tips
- GPU settings are passed via CLI flags; avoid hard-coding device IDs.
- Keep large artifacts out of Git; prefer the existing results directories for outputs.
