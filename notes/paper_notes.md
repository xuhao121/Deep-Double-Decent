# Paper notes: settings and ambiguities

## Settings used (source in the paper)

| Setting | Value | Source |
|---|---|---|
| ResNet18 | pre-activation, stage widths [k, 2k, 4k, 8k], strides [1, 2, 2, 2], k = 64 standard | App. B.1 |
| 5-layer CNN | 4 x Conv-BN-ReLU-MaxPool + FC, widths [k, 2k, 4k, 8k], MaxPool [1, 2, 2, 8], 1,558,026 params at k = 64 | App. B.1 |
| Adam | lr 1e-4 constant, other arguments default, 4K epochs | Sec. 4, App. B.2 |
| SGD | lr0 0.1, lr(t) = lr0 / sqrt(1 + floor(t/512)), no momentum, 500K steps | Sec. 4, App. B.2 |
| Batch size | 128 | App. B.2 |
| Augmentation | RandomCrop(32, padding=4) + RandomHorizontalFlip; all views share the (noisy) label | App. B.2 |
| Label noise | with probability p a uniformly random incorrect label, sampled once | Sec. 4 |
| Test error | clean test set, except Fig. 1 (noisy distribution) | Sec. 4 |

## Ambiguities and inconsistencies found

1. App. B.4: Fig. 11a models trained for "500K epochs"; Sec. 4 says SGD runs 500K gradient steps.
2. App. B.1: Transformers trained "for 80 gradient steps"; Sec. 4 says 80K.
3. Fig. 10 caption "(width=128)": unclear whether it refers to the CNN only or to both models.
4. Fig. 1 plots test error on the noisy distribution; all other figures use the clean test set.
5. App. B.1 describes the ResNet as "4 ResNet blocks, each consisting of two BatchNorm-ReLU-Convolution layers", which does not add up to 18 layers; the authors' resnet18k.py is authoritative.
6. Not specified: input normalisation, whether the last partial batch is dropped, the exact widths of the three regimes in Fig. 9.
7. The authors' notebook `intro_resnet_plot.ipynb` (which draws Fig. 1) converts clean test error e to noisy test
   error with `1 - (1-p)(1-e) + e*p/9`. The correct expectation under uniform incorrect-label noise is
   `p + e(1 - 10p/9)`; the notebook's term `+ e*p/9` should be `- e*p/9`. Monte-Carlo check (K=10, p=0.15,
   e=0.30): simulated 0.4004, correct formula 0.4000, notebook 0.4100. The published Fig. 1 test curve is
   therefore too high by 2pe/9 (about 1 percentage point). The raw data itself stores clean error.
8. Raw data: README of the authors' repository links each experiment to `gs://hml-public/dd/<folder>/`
   (`Mlist`, `ks` pickles). The index meaning of `Mlist` is documented only for the ResNet folders; the
   CNN folders are undocumented; there is no raw data for Fig. 21 (weight decay) or Fig. 28 (ensembles);
   the NLP notebooks read local CSV files that are not in the repository. The README row for the CIFAR-100
   MCNN folder says "0, 10, 20 % label noise and data-aug", but the folder (`pct-cifar100-mcnn-p0-sgd-noaug-reps`)
   holds 5 nearly identical curves, i.e. repetitions without noise or augmentation, as its name says.
9. Inferred from the raw data (not stated in the paper or README), and confirmed two ways: the CNN folders
   with augmentation (`cifar10-mcnn-p20-sgd*`, `cifar10-mcnn-p10-sgd`) store test error against NOISY test
   labels, unlike the ResNet and no-augmentation CNN folders (`python -m analysis.orig_data identify`).
   (a) At the smallest width, which underfits, train error ~= test error in all 18 entries (train - test
   between -0.014 and +0.012 at k=1-3), whereas a clean test set would give p(1 - 10*test/9), about +0.08
   at 20 % noise and +0.05 at 10 %; the no-augmentation folder shows that clean pattern (20 % noise, k=1:
   train 0.637, test 0.572, i.e. +0.065 vs +0.073 predicted). (b) Converted to clean error, the authors'
   error at k=64 agrees with ours to within 0.014 (20 % noise, n = 12.5k / 25k / 50k: 0.294 / 0.272 / 0.261,
   mean of 5 repetitions, vs ours 0.308 / 0.277 / 0.261; 10 % noise, n = 50k: 0.187 vs 0.183); unconverted
   it is about 0.14 higher (n = 50k, k=64: 0.40 vs ours 0.26). Consequence: Fig. 11a/12, and the
   with-augmentation curves of Fig. 5 if they come from these folders, most likely show noisy test error
   while the captions do not say so (check the y-axis values in the paper's figures).
10. The noisy-error helper with the wrong-sign term (item 7) also appears in `intro_ocean_dynamics.ipynb`
   (`plot_dynamics`, parameter p), not only in the notebook that draws Fig. 1.
11. Identified from the raw data (`identify`): the sample-wise folders hold training-set sizes
   n = 12.5k / 25k / 50k (at 20 % noise, 5 repetitions; at 10 % noise, 1 run), i.e. the 2x and 4x comparisons
   of Fig. 11a; raw data for the many sizes of the Fig. 12 heat map is not published. The 20 % repetition
   folders rep2-rep5 log 975 points instead of 1952 in the first one, so they were logged half as often or
   trained for fewer steps; their interpolation thresholds are slightly larger (n = 50k: 46 vs 42), which
   fits shorter training, but the training length cannot be read off the data. The 10 % folder logs 1421
   points, the CIFAR-100 folder 1456.

## Our choices for unspecified details

(see README, "Implementation choices not specified in the paper")
