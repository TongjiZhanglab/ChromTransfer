# ChromTransfer: Cross-Species TF Binding Site Prediction

ChromTransfer predicts transcription factor (TF) binding sites in a target species by training on source-species ChIP-seq data and performing genome-wide inference via cross-species transfer.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Download Required Data from Figshare](#2-download-required-data-from-figshare)
3. [Configure User-Defined Parameters](#3-configure-user-defined-parameters-configpy)
4. [Data Preprocessing](#4-data-preprocessing-preprocesspy)
5. [Train Model in Source Species](#5-train-model-in-source-species-trainpy)
6. [Genome-Wide Prediction in Target Species](#6-genome-wide-prediction-in-target-species-predictpy)

---

## 1. Prerequisites

### 1.1 Command-line tools

Install the following tools and ensure they are available in your `PATH`:

| Tool | Purpose |
|------|---------|
| `bedtools` | Intersect peak files with genome bins |
| `macs2` | Call peaks from predicted scores |
| `wigToBigWig` | Convert WIG output to BigWig format |

### 1.2 Python environment

Python ≥ 3.9 is recommended. Install dependencies:

```bash
conda create -n chromtransfer python=3.10 -y
conda activate chromtransfer
pip install torch pandas numpy scikit-learn tqdm scanpy h5py
```

---

## 2. Download Required Data from Figshare

Download data/ from Figshare (https://doi.org/10.6084/m9.figshare.31972113) and place the files under /data in your ChromTransfer repository directory. 
```
cd ${YOUR_PATH_TO_ChromTransfer}/data
rm -r * ### delete the existing empty folders in data directory from this repository

# download cobinding_TF_source.tar.gz
curl -L -o cobinding_TF_source.tar.gz "https://ndownloader.figshare.com/files/63597297"
tar -xzvf cobinding_TF_source.tar.gz

# download DNA.tar.gz
curl -L -o DNA.tar.gz "https://figshare.com/ndownloader/files/63597807"
tar -xzvf DNA.tar.gz

# download FUNCODE.tar.gz
curl -L -o FUNCODE.tar.gz "https://figshare.com/ndownloader/files/63598410"
tar -xzvf FUNCODE.tar.gz

# download regions.tar.gz
curl -L -o regions.tar.gz "https://figshare.com/ndownloader/files/63600303"
tar -xzvf regions.tar.gz

# download Regulatory.tar.gz
curl -L -o Regulatory.tar.gz "https://figshare.com/ndownloader/files/63600441"
tar -xzvf Regulatory.tar.gz
```
The expected directory structure is:

```
${YOUR_PATH_TO_ChromTransfer}/
└── data/
    ├── DNA/
    │   ├── hg38_DNA_500bpBin50bpStep_region.h5
    │   └── mm10_DNA_500bpBin50bpStep_region.h5
    ├── FUNCODE/
    │   ├── hg38_FUNCODE_avgScore_50050.pkl
    │   └── mm10_FUNCODE_avgScore_50050.pkl
    ├── Regulatory/
    │   ├── TF_h5Column_mask_dict_prepared.json
    │   ├── hg38_Reg_signal_matrix.hdf5
    │   └── mm10_Reg_signal_matrix.hdf5
    ├── cobinding_TF_source/
    │   ├── CAP_SELEX/
    │   ├── ChIP_Atlas/
    │   ├── STRING/
    │   ├── cobindingTF_chromatinContext_ls.txt
    │   ├── ensemblProteinID_2_geneName_hg38.txt
    │   └── ensemblProteinID_2_geneName_mm10.txt
    └── regions/
        ├── hg38_500_50_noblack_k36_noN_window_regionNumber.bed
        └── mm10_500_50_noblack_k36_noN_window_regionNumber.bed
```

## (Optional) Download Demo Files to Get a Quick Start of ChromTransfer:
Download demo.tar.gz from Figshare (https://doi.org/10.6084/m9.figshare.31972113) and place it under your ChromTransfer repository directory. 
```
cd ${YOUR_PATH_TO_ChromTransfer}
rm -r demo ### delete the existing empty demo directory from this repository
curl -L -o demo.tar.gz "https://figshare.com/ndownloader/files/63608829"
tar -xzvf demo.tar.gz
```
The expected directory structure is:

```
${YOUR_PATH_TO_ChromTransfer}/
└── demo/
    ├── 1.preprocess_data/
    ├── 2.train_model/
    └── 3.predict/
```
---

## 3. Configure User-Defined Parameters (`config.py`)

Open `config.py` and edit the parameters before running any script.

### 3.1 Necessary parameters

| Parameter | Description |
|-----------|-------------|
| `YOUR_PATH_TO_ChromTransfer` | Absolute path to the ChromTransfer repository directory (where you placed this repository) |
| `output_dir` | Directory where the model checkpoints and all output files will be saved |
| `source_species` | Species used for training. Either `"mm10"` or `"hg38"` |
| `target_species` | Species used for prediction. Either `"hg38"` or `"mm10"` |
| `peak_file_source` | Path to the TF ChIP-seq peak file in the source species (BED format) |

### 3.2 Optional parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `peak_file_target` | `""` | Path to the TF ChIP-seq peak file in the target species (BED). Leave empty if unavailable; cross-species prediction will still run, but held-out evaluation on target chr2 will be skipped |
| `tf` | `"SOX2"` | Name of the TF to predict (uppercase) |
| `model` | `"ChromTransfer-Reg"` | Model variant: `"ChromTransfer-Base"`, `"ChromTransfer-Cons"`, or `"ChromTransfer-Reg"` |
| `test_chromosome` | `"chr2"` | Chromosome held out for testing |
| `valid_chromosome` | `"chr1"` | Chromosome from which validation regions are sampled. The rest regions of this chromosome will also be included in training dataset together with other chromosomes. |
| `valid_chromosome_region_num` | `100000` | Number of regions randomly sampled from `valid_chromosome` for validation |
| `max_epoch_num` | `100` | Maximum number of training epochs |
| `random_seed` | `9999` | Random seed for reproducibility |
| `gpu` | `"0"` | GPU device ID |
| `batch_size` | `1024` | Batch size for training |
| `lr` | `1e-4` | Learning rate |
| `num_workers` | `32` | Number of DataLoader workers for training |
| `predict_batch_size` | `4096` | Batch size for inference |
| `predict_num_workers` | `32` | Number of DataLoader workers for inference |
| `threshold_bin_width` | `0.0001` | Resolution of the score-to-FDR mapping table (smaller = more accurate but slower) |
| `threshold_FDR_cutoff` | `0.2` | FDR cutoff used to call peaks |

---

## 4. Data Preprocessing (`preprocess.py`)

This step intersects the source-species peak file with genome bins, assigns binary labels, and splits data into balanced train/validation/test sets.

```bash
python preprocess.py

### If you want to try our demo data for preprocessing, you can set `self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/1.preprocess_data"` in `config.py`, then run the command above.
```

**Outputs** (saved to `output_dir`):

| File | Description |
|------|-------------|
| `<source_species>_label.bed` | Full label BED file |
| `<source_species>_train.pkl` | Pickled train set |
| `<source_species>_valid.pkl` | Pickled validation set |
| `<source_species>_test.pkl` | Pickled test set |
| `<source_species>.<valid_chromosome>_random<N>.txt` | Validation region list (used during training) |
| `<source_species>.<test_chromosome>.txt` | Test region list |
| `<source_species>.chrOthers_epoch{1,2,...}.txt` | Epoch-wise balanced train files (positives + negatively-downsampled negatives) |

---

## 5. Train Model in Source Species (`train.py`)

Train ChromTransfer on the source species using the preprocessed label files. The model with the best validation AUPRC is saved. Early stopping triggers if AUPRC does not improve for 5 consecutive epochs.

```bash
python train.py

### If you want to try our demo data for training, you can set `self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/2.train_model"` in `config.py`, then run the command above.
```

**Outputs** (saved to `output_dir`):

| File | Description |
|------|-------------|
| `best_model.pth` | Best model checkpoint (highest validation AUPRC) |
| `train_loss.txt` | Per-epoch training loss and accuracy |
| `valid_loss_<source_species>.txt` | Per-epoch validation metrics (loss, AUC, AUPRC, etc.) |
| `predictions_<source_species>Chr2.txt` | Model score on source-species chr2 (used by `predict.py` for FDR calibration) |
| `test_loss_<source_species>.txt` | Test set evaluation metrics |
| `predictions_<target_species>Chr2.txt` | Model score on target-species chr2 (only if `peak_file_target` is set) |
| `test_loss_<target_species>.txt` | Target-species test evaluation metrics (only if `peak_file_target` is set) |

---

## 6. Genome-Wide Prediction in Target Species (`predict.py`)

Use the trained model to score all genomic bins in the target species, calibrate scores to FDR using the source-species held-out chromosome, and call peaks.

```bash
python predict.py

### If you want to try our demo data for prediction, you can set `self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/3.predict"` in `config.py`, then run the command above.
```

**Outputs** (saved to `output_dir`):

| File | Description |
|------|-------------|
| `predictions_<target_species>.txt` | Raw model scores for all bins in target genome |
| `<source_species>_chr2_threshold_FDR.csv` | Score-to-FDR mapping table derived from source chr2 |
| `<target_species>_total_threshold_FDR.csv` | FDR-annotated predictions for the full target genome |
| `FDR_<target_species>_50bp_bins.bed` | FDR signal per 50 bp bin (BED format) |
| `FDR_<target_species>_50bp_bins.wig` | FDR signal per 50 bp bin (WIG format) |
| `<target_species>_narrowPeak.bed` | Called peaks (MACS2, FDR ≤ `threshold_FDR_cutoff`) |

---

## Quick Start

```bash
# Step 1 — edit config.py with your paths and TF of interest, then:

# Step 2 — preprocess
python preprocess.py

### If you want to try our demo data for preprocessing, you can set `self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/1.preprocess_data"` in `config.py`, then run the command above.

# Step 3 — train
python train.py

### If you want to try our demo data for training, you can set `self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/2.train_model"` in `config.py`, then run the command above.

# Step 4 — genome-wide prediction
python predict.py

### If you want to try our demo data for prediction, you can set `self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/3.predict"` in `config.py`, then run the command above.

```
