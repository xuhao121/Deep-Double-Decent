# Sanity-check log (step 2 of the plan)

| # | Check | Command | Result | Date | By |
|---|---|---|---|---|---|
| 1 | Parameter counts | `python scripts/sanity_checks.py params` | | | |
| 2 | Label noise | `python scripts/sanity_checks.py noise` | | | |
| 3 | Augmentation (see aug_grid.png) | `python scripts/sanity_checks.py aug` | | | |
| 4 | Overfit 1000 images | `python scripts/sanity_checks.py overfit` | | | |
| 5 | CNN clean baseline > 90 % | R10 run `cnn_c10_p00_ns0_k064_s0_n50000_aug1_wd0` | | | |
| 6 | Resume after Ctrl-C | test run, then `sanity_checks.py csv` | | | |
| 7 | Error definition (0-1, not loss) | `sanity_checks.py csv` | | | |
| 8 | Complete logs | `sanity_checks.py csv` | | | |
