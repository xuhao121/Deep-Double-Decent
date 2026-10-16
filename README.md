# Deep Double Descent — Reproducibility Study (Replication Track)

Reproduction of Nakkiran et al., *Deep Double Descent: Where Bigger Models and More Data Hurt*
(ICLR 2020, [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)).

- **Group:** _number_ — _member 1_, _member 2_
- **Track:** Replication (the authors released model definitions, plotting notebooks and raw data, not their
  training code; the training pipeline here is re-implemented from the paper's Appendix B)

| Claim | What we test | Figures | Owner |
|---|---|---|---|
| C1 model-wise | test error peaks at the interpolation threshold (ResNet18 width sweep) | Fig. 1, 2, 4b | member 1 |
| C2 epoch-wise | large models: test error falls, rises, falls again over training | Fig. 9 | member 1 |
| C3 sample-wise | more data shifts the peak right; near-critical models do not benefit from 2-4x data | Fig. 11a, 12 | member 2 |
| C4 EMC | label noise, augmentation and more samples move the threshold and the peak | Fig. 5, 21 | member 2 |

## Code and data from the authors

1. **Model definitions** - `src/models/resnet18k.py` and `src/models/mcnn.py` are the authors' files from
   <https://gitlab.com/harvard-machine-learning/double-descent> (commit `0d653296`), copied unmodified
   (only a provenance header added). Verified: the CNN has 1,558,026 parameters at k=64, as in the paper.
2. **Raw results** - stored by the authors in the public bucket `gs://hml-public/dd/<folder>/` as two pickles,
   `Mlist` and `ks` (format documented in `analysis/orig_data.py`). Download them with
   `bash scripts/download_orig_data.sh`; they are then overlaid on our figures automatically.
   The pickles are not committed (`orig_data/` is in .gitignore); each ResNet folder holds 64 widths x 4000 epochs per metric.
3. **CIFAR-10/100** - downloaded automatically by `scripts/prepare_data.py` (torchvision).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate          # Python 3.11 or 3.12
# B200 (Blackwell) needs a PyTorch build with CUDA >= 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/prepare_data.py      # CIFAR-10/100 -> data/*.npy
python scripts/make_runs.py         # runs.csv: all experiments R1-R13 (except R3, R6)
python scripts/make_labels.py       # noisy labels + subset indices -> labels/ (commit them)
```

All paths can be redirected with environment variables (`DD_DATA_DIR`, `DD_LABELS_DIR`, `DD_OUTPUTS_DIR`,
`DD_RESULTS_DIR`, `DD_RUNS_CSV`, `DD_TIMING_JSON`), e.g. to put `outputs/` on a larger disk.

## Workflow

**1. Sanity checks** (all must pass before the real runs; record results in `notes/sanity_checks.md`)

```bash
python scripts/sanity_checks.py params     # CNN k=64 must have exactly 1,558,026 parameters
python scripts/sanity_checks.py noise      # flip rate, never the true class, uniform wrong class
python scripts/sanity_checks.py aug        # notes/aug_grid.png
python scripts/sanity_checks.py overfit    # ResNet18 k=64 fits 1000 images to 0 train error
python scripts/sanity_checks.py test-runs  # runs_test.csv: two tiny runs for resume/log checks
python scripts/make_labels.py --runs-csv runs_test.csv
python -m src.train --run test_r18_c10_p15_ns0_k004_s0 --runs-csv runs_test.csv   # Ctrl-C once mid-way
python -m src.train --run test_r18_c10_p15_ns0_k004_s0 --runs-csv runs_test.csv   # resumes
python scripts/sanity_checks.py csv --run test_r18_c10_p15_ns0_k004_s0 --runs-csv runs_test.csv
```

Check 5 (CNN k=64, 0 % noise, augmentation reaches > 90 % test accuracy) is the R10 run
`cnn_c10_p00_ns0_k064_s0_n50000_aug1_wd0`.

**2. Measure speed on one MIG slice** (fills `timing.json`, used for estimates and scheduling)

```bash
nvidia-smi -L                                        # lists the MIG UUIDs
export CUDA_VISIBLE_DEVICES=MIG-xxxxxxxx
python scripts/pilot_timing.py single --model resnet18k --k 2 8 16 32 64 --epochs 3
python scripts/pilot_timing.py single --model mcnn --k 4 16 64 --steps 2000
python scripts/pilot_timing.py concurrent --model resnet18k --k 8 16 --procs 2 3
python scripts/pilot_timing.py concurrent --model mcnn --k 4 16 --procs 2 3
python scripts/pilot_timing.py merge
python scripts/make_runs.py --update-estimates
```

`merge` prints how much slice throughput 2 or 3 concurrent light runs give; pass the best value to
`assign_jobs.py --light-lanes`.

**3. Launch** (each member on their own machine)

```bash
python scripts/assign_jobs.py --owner m1        # member 2: --owner m2
bash scripts/launch_tmux.sh jobs/m1/lanes.tsv m1
tmux attach -t dd-m1                            # Ctrl-b n / Ctrl-b p: next/previous lane, Ctrl-b d: detach
```

Every run resumes automatically from its checkpoint. Logs: `logs/<run_id>.log`.

**4. Monitor**

```bash
python scripts/status.py --owner m1             # per experiment: done / running / paused / crashed / todo
python scripts/status.py --exp R1 R2 --detail   # per run: progress, current errors, ETA
```

Commit `results/` (small CSV + JSON files) daily so both members see all results.

**5. After P0: add the refinement and ensemble widths, re-plan**

```bash
python scripts/make_runs.py --r3 7 9 11 13 18 --r6 4 6 8 10 12 2 32 64   # widths chosen from R1/R2
python scripts/make_labels.py
bash scripts/stop_workers.sh                    # each run checkpoints at the end of its epoch
python scripts/assign_jobs.py --owner m1
bash scripts/launch_tmux.sh jobs/m1/lanes.tsv m1
```

**6. Move work to the idle machine** when member 2's queue is finished

```bash
# on member 1's machine
python scripts/rebalance.py --from m1 --to m2 --exp R5 R6 R7 --fraction 0.5
git add runs.csv && git commit -m "rebalance" && git push
# on member 2's machine
git pull && python scripts/make_labels.py && python scripts/assign_jobs.py --owner m2 \
  && bash scripts/launch_tmux.sh jobs/m2/lanes.tsv m2
```

Member 1's workers skip the moved runs automatically (`train.py --owner`).

**7. The authors' raw results** (for the dashed "paper" curves and the paper columns of T1/T2)

```bash
bash scripts/download_orig_data.sh              # gsutil if installed, otherwise curl
python -m analysis.orig_data check              # which folders / figures are available
python -m analysis.orig_data identify           # re-checks the CNN folders' settings (prints the evidence)
```

The meaning of the `Mlist` entries is documented by the authors only for the two ResNet folders
(repetitions; noise 0/10/20 %). For the 5-layer-CNN folders it was worked out from the data with `identify`;
the results are already in `FOLDERS` in `analysis/orig_data.py`, each with its evidence:

- **Test-error scale.** At the smallest width that trained, which underfits, a test set with noisy labels
  gives train error ~= test error, a clean one gives train - test ~= p(1 - K*test/(K-1)). The CNN folders
  with augmentation (p20, p10) store error against *noisy* test labels; it is converted to clean error
  before any comparison. The no-augmentation and CIFAR-100 folders store clean error.
- **Entry -> setting.** Each entry's final training-error curve is compared with those of our finished runs;
  training-set size and noise level move the interpolation threshold, so the closest curve identifies the
  entry. The p20 and p10 entries are n = 12.5k / 25k / 50k; the no-augmentation entries 0 / 10 / 20 % noise.

`identify --write` saves clear-cut results to `analysis/orig_folders.json`, which overrides `FOLDERS`; it is
only needed if the folders change. If a download fails, digitise the figure (WebPlotDigitizer) into
`analysis/orig_digitized/<figure>.csv` instead. Fig. 21 and Fig. 28 have no published raw data.

**8. Figures and tables**

```bash
python -m analysis.make_figures all             # or: resnet | cnn | a1 a3 t1 ...
```

With the raw data present, the figures show the paper's curves dashed in the same colours (only over the
widths we trained), A2 adds the paper's heat map and an ours-minus-paper map, and T1/T2 add `paper_*`
columns and `d_*` differences (the quantitative criterion in `notes/success_criteria.md`). The paper's
threshold and peak are read off at *our* widths, so `d_peak_steps` counts steps of our grid; `*_dense`
columns give the paper's values on its own finer grid, and `paper_final_k` the width its final error is from.

To develop the analysis before real results exist, generate synthetic ones (never mixed with real results):

```bash
DD_RESULTS_DIR=/tmp/fake/results DD_OUTPUTS_DIR=/tmp/fake/outputs python tests/make_fake_results.py
DD_RESULTS_DIR=/tmp/fake/results DD_OUTPUTS_DIR=/tmp/fake/outputs python -m analysis.make_figures all
rm -rf figures tables                           # afterwards
```

**9. Moving to another machine** (e.g. full GPUs instead of MIG slices)

Runs resume from their checkpoints on any GPU; `results/<run_id>.meta.json` records host and device of every
session. Either keep both machines and move only not-started runs to a new owner tag:

```bash
python scripts/rebalance.py --from m1 --to m1b --exp R4 R5 R7 --fraction 0.6   # old machine; commit runs.csv
DD_TIMING_JSON=timing_2gpu.json python scripts/assign_jobs.py --owner m1b --lanes-per-device 8   # new machine
bash scripts/launch_tmux.sh jobs/m1b/lanes.tsv m1b
```

or move everything: `bash scripts/stop_workers.sh` on the old machine (wait until no process and no
`outputs/*/RUNNING` is left), copy `runs.csv timing.json labels outputs results notes` to the new machine,
measure it (`pilot_timing.py` with `DD_TIMING_JSON=timing_2gpu.json`, `concurrent --procs 4 8 12`), then
`assign_jobs.py --owner m1 --lanes-per-device N` and `launch_tmux.sh`. On full GPUs use `--lanes-per-device`
(many runs per GPU); `--light-lanes/--heavy-slices` are for MIG slices.

## Experiments (`runs.csv`)

| Block | Prio | Setting | Runs | Report |
|---|---|---|---|---|
| R1 | P0 | ResNet18, 15 % noise, 16 widths, seed 0 | 16 | Fig. 1, 2 |
| R2 | P0 | ResNet18, 20 % noise, 16 widths, seed 0 | 16 | Fig. 9, 4b |
| R3 | P0 | ResNet18, 15 % + 20 %, extra widths near the peak | ~10 | denser peak |
| R4 | P1 | ResNet18, 0 % + 10 % noise, 11 widths | 22 | Fig. 4b |
| R5 | P1 | ResNet18, 15 %, seeds 1-4, same noisy labels | 64 | Fig. 1 error bars |
| R6 | P2 | ResNet18, 15 %, seeds 1-4, independent noise draws | 32 | Fig. 28 |
| R7 | P2 | ResNet18 k=64, 20 %, SGD + momentum 0.9 | 1 | App. E.1 |
| R8 | P0 | CNN, 20 %, augmentation, n ∈ {6250, 12500, 25000, 50000} | 56 | Fig. 11a, 12 |
| R9 | P0 | CNN, 20 %, no augmentation | 14 | Fig. 5 |
| R10 | P1 | CNN, {0 %, 10 %} × {aug, no aug} | 56 | Fig. 5 |
| R11 | P1 | CNN, 10 %, augmentation, weight decay 5e-4 | 14 | Fig. 21 |
| R12 | P1 | as R8, seeds 1-4 (new init, subset and noise draw) | 224 | Fig. 11a error bars |
| R13 | P2 | CNN, CIFAR-100, 0 %, no augmentation, 1M steps, 5 seeds | 70 | Fig. 7 |

Shared settings (paper, Appendix B): batch 128, cross-entropy, PyTorch default initialisation, no
weight decay or dropout unless stated. ResNet18: Adam lr 1e-4 (constant), 4000 epochs. CNN: SGD
lr 0.1 · 1/√(1 + ⌊t/512⌋), no momentum, 500k steps. Augmentation: RandomCrop(32, padding=4) +
horizontal flip. Label noise: with probability p a uniformly random *wrong* label, drawn once.

## Outputs of a run

| File | Content |
|---|---|
| `results/<run_id>.csv` | one row per evaluation: epoch, step, time, lr, train error/loss on the noisy training labels, error on clean vs flipped samples, fraction of noisy labels memorised, clean test error/loss |
| `results/<run_id>.meta.json` | full config, git commit, versions, device, parameter count, label-file hash, realised noise rate, training time, final metrics |
| `outputs/<run_id>/test_preds.npy` | predicted test classes at every evaluation (uint8, 255 = not written) |
| `outputs/<run_id>/train_preds.npy` | predicted training classes at ~20 log-spaced evaluations |
| `outputs/<run_id>/final.pt` | final weights; `ckpt.pt` exists only while the run is unfinished |

Evaluations: ResNet every epoch up to 100, every 5 up to 500, every 10 up to 1000, every 50 after
(290 points); CNN 100 log-spaced steps.

## Implementation choices not specified in the paper

- Normalisation: CIFAR-10 mean/std constants of kuangliu/pytorch-cifar; CIFAR-100: training-set statistics.
- The last partial batch of an epoch is kept (DataLoader default `drop_last=False`).
- Mixed precision: bf16 autocast on GPU (`--no-amp` for an fp32 comparison run).
- Noise levels are nested for a fixed noise seed (samples flipped at 15 % ⊂ those flipped at 20 %);
  training subsets are nested (first n entries of a fixed permutation) and inherit the full-set noise.
- Ensembles (Fig. 28): plurality vote; ties go to the lowest-seed model's prediction.
- Fig. 1 of the paper plots test error on the *noisy* distribution; we report clean test error, which is
  also what the authors' raw data stores. Their notebook converts clean to noisy error with
  `1-(1-p)(1-e) + e*p/9`; the correct expectation is `p + e(1 - 10p/9)` (verified by simulation), so the
  published Fig. 1 is too high by `2pe/9` (about 1 point at p=0.15). Both versions are in `analysis/metrics.py`.

## Repository layout

```
src/        config, data pipeline (all on GPU), label noise, schedules, training loop, evaluation
scripts/    data, run list, labels, sanity checks, timing, job assignment, workers, status, rebalancing
analysis/   result loading, metrics, figures/tables, original-data loader (template)
tests/      synthetic results for dry-running the analysis
notes/      paper notes, sanity-check log, pre-registered success criteria
results/    per-run CSV + meta.json (committed)     labels/  noisy labels, subsets (committed)
figures/    report figures (committed)              tables/  report tables (committed)
```

## Credits

Model definitions (included unmodified) and raw results: Nakkiran et al.,
<https://gitlab.com/harvard-machine-learning/double-descent> - the repository has no license file; the
code is included with attribution for this academic reproduction. Their ResNet is adapted from
<https://github.com/kuangliu/pytorch-cifar>.
