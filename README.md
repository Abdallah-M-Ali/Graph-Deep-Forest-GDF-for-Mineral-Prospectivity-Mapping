# Graph Deep Forest (GDF) for Mineral Prospectivity Mapping

**Authors:** Abdallah M. Mohamed Taha, Gang Liu etc. 
**Corresponding manuscript:** *"graph deep forest for spatial learning in mineral prospectivity mapping"* – submitted to *Computers & Geosciences*.

This repository provides the official implementation of the **Graph Deep Forest (GDF)** framework, which combines graph‑based random forest message passing with a cascade deep forest regressor to produce continuous mineralization probability maps.

---

## 📖 Overview of the Method

1. **Graph construction** – Each pixel in the input raster becomes a node. Edges connect each pixel to its 8 neighbours (King’s move). Node features are the multi‑band evidential values.
2. **Multi‑stage message passing** – For each head (stage), a random forest is trained to predict a node’s own features from the concatenated features of its neighbours. The forest’s predictions are used to **update** node features via an alpha‑weighted residual connection. This is repeated for `num_heads` stages, producing `num_heads` different representations of the same graph.
3. **Cascade deep forest regressor** – The `num_heads` feature matrices are fed as alternating inputs to a cascade of random forests (the `DFCascadeRegressor`). Each layer of the cascade concatenates the output of the previous layer with the next head’s features in a rolling manner. The final output is a continuous value (0–1) representing the estimated mineralization probability.
4. **Evaluation** – The model is evaluated test‑set AUC, MSE, and overall accuracy.

---

## 📁 Data Requirements

### Input raster (`features.tiff`)
- **Format:** GeoTIFF, any bit depth (float or integer).
- **Bands:** Each band is one evidential layer (e.g., geochemistry, geophysics, remote sensing).
- **NoData values:** Should be set to `NaN` or `0` (the code replaces `NaN` with `0`).
- **Spatial reference:** Must match the shapefile’s CRS.

### Training / testing shapefiles (`training70.shp`, `testing30.shp`)
- **Format:** ESRI Shapefile (point or polygon – if polygon, they will be rasterized).
- **Attribute field:** Must contain an integer field named **`raster`** with the following **label convention**:
  - `2` = mineralized (ore deposit)
  - `1` = non‑mineralized (barren)
  - (Any other value or missing will be ignored)

> **Why 2 and 1?**  
> The code internally converts these to binary labels using `y = truth - 1`, so `2 → 1` (positive) and `1 → 0` (negative). This is a common practice when shapefiles originally have `1` for background and `2` for deposit. If your shapefile already uses `0/1`, change the `-1` in `dataFitting` to `0` – but the README follows your current implementation.

### Example attribute table

| FID | geometry | raster |
|-----|----------|--------|
| 0   | Point    | 2      |
| 1   | Point    | 1      |
| 2   | Point    | 2      |

---

## 🛠 Installation

### Option 1: Using `pip`
```bash
git clone https://github.com/yourusername/GDF-Mineral-Prospectivity.git
cd GDF-Mineral-Prospectivity
pip install -r requirements.txt
```

### Option 2: Using Conda (recommended for GDAL)
```bash
conda env create -f environment.yml
conda activate gdf-mpm
```

---

## 🚀 Workflow – Step by Step

### 1. Prepare your data
Place the following files inside the `data/` folder:
- `features.tiff`
- `training70.shp` (and its auxiliary files: `.shx`, `.dbf`, `.prj`)
- `testing30.shp` (same auxiliary files)

```bash
python scripts/train_and_evaluate.py
```

Results are saved to `results/hyperparameter_tuning.csv`. The script will print the top 5 parameter combinations sorted by MSE and AUC.

### 2. Train the final model and generate the full prospectivity map
Use the best parameters from the tuning step (or the default values if you skip tuning).
the tst file includes all the process for model tuning, important features analysis, and the producing MPM

```bash
python scripts/predict_full_map.py --n_trees 300 --alpha 0.3 --n_estimators 8
```

The script will:
- Build the graph from the full raster.
- Run multi‑stage message passing (3 heads by default).
- Train a cascade forest regressor on all labeled nodes (from `training70.shp`).
- Predict mineralization probabilities for **every node** in the raster.
- Save a GeoTIFF file `results/GDF_prospectivity.tiff`.



---

## 📂 Repository Structure (with explanations)

```
GDF-Mineral-Prospectivity/
├── README.md                    # This file
├── LICENSE                      # MIT license
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── setup.py                     # Editable install
├── data/                        # Place your input files here (not tracked by git)
│   ├── features.tiff
│   ├── training70.shp
│   └── testing30.shp
├── src/
│   ├── __init__.py
│   ├── Data_preprocessing.py    # Raster loading, reshaping, shapefile rasterization
│   └── test.py               # training the whole GDF model and producing prosepctivity map
│   └── Graph/
│       ├── __init__.py
│       ├── DF.py                # Cascade forest regressor (DFCascadeRegressor)
│       └── GDF.py               # MineralGraphBuilder (graph + message passing)
```

---

## ⚙️ How the Code Handles Labels (Important for Users)

- **Shapefile attribute `raster`** must contain `2` (mineralized) and `1` (non‑mineralized).
- Inside `MineralGraphBuilder.dataFitting()` (and `.dataFitting_aug()`), the line `y = truth[idx] - 1` converts `2→1` and `1→0`.
- Internally, the model treats `1` as the positive class (mineralized) and `0` as negative.
- The regressor outputs continuous values. For computing **overall accuracy**, the script thresholds at 0.5: `(y_pred >= 0.5).astype(int)`.

If your shapefile already uses `1` for mineralized and `0` for non‑mineralized, modify the line to `y = truth[idx]` (i.e., remove the `-1`). The README assumes you follow the `2/1` convention.

---

## 🔬 Reproducibility

All random seeds are fixed:
- `random_state=19` in the graph message‑passing forests.
- `random_state=42` in the cascade forest regressor.
- `np.random.seed(42)` is not explicitly set but all sklearn estimators use the given random states.

To exactly reproduce the numbers reported in the manuscript, use the hyperparameters listed in **Table 2** of the paper (typically `n_trees=300`, `alpha=0.3`, `n_estimators=8`, `num_heads=3`).

---

## 📝 Citation

If you use this code or the GDF method in your research, please cite our *Computers & Geosciences* article:

```
[Mohamed Taha A. M., etc.], (2026). graph deep forest model for spatial learning in mineral prospectivity mapping. Computers & Geosciences, Vol. XX, Article XXXXX.
```

BibTeX entry will be added upon publication.

---

## ❓ Troubleshooting

| Issue | Likely solution |
|-------|------------------|
| `gdal` import error | Use Conda environment – `conda install gdal` |
| `MemoryError` during full raster prediction | The script includes a batch prediction fallback. Reduce `batch_size` in `predict_full_map.py`. |
| Position mismatch in output raster | Ensure `MineralGraphBuilder.prepare_rf_training_data_with_heads()` builds `position` using **sorted nodes**. The provided code does this. |
| Labels not recognised | Check that the shapefile attribute field is named exactly `raster` (case‑sensitive) and contains integers 2 and 1. |

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the first author at [ali.abdallah.0992@gmail.com].

---

**Last updated:** May 2026  
**License:** GNU

---

This README now explains the **label convention (2/1)** explicitly, includes author names, and walks through the entire data processing pipeline. You can replace the placeholder author names and email before uploading to GitHub.
