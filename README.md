# Pulmonary Chest X-Ray (Tuberculosis) Detection — CNN

A CNN-based classifier that predicts whether a chest X-ray is **Normal** or
shows features consistent with **Tuberculosis**, trained on the Kaggle
dataset `kmader/pulmonary-chest-xray-abnormalities` (Montgomery + Shenzhen
collections). Originally developed in `pulmonary-chest-x-ray.ipynb`; this
folder turns it into a runnable local project (PyCharm-ready) with a
Streamlit demo app.

> ⚠️ **Disclaimer**: This is an academic/research prototype (see
> `SRS_Pulmonary_CXR_CNN.docx`). It is **not** a certified medical device and
> must never be used for real diagnostic or treatment decisions.

## Project structure

```
project/
├── app.py              # Streamlit web app (upload an X-ray, get a prediction)
├── predict.py           # CLI single-image inference script
├── requirements.txt
├── models/
│   ├── best_tb_cnn.keras            # best checkpoint (highest val AUC)
│   └── tb_chest_xray_cnn.keras      # final model saved at end of training
└── README.md
```

## Setup in PyCharm

1. **Open the project**: `File -> Open...` and select this `project` folder.
2. **Create a virtual environment** (PyCharm will usually prompt you, or do
   it manually):
   - `File -> Settings -> Project -> Python Interpreter -> Add Interpreter ->
     Add Local Interpreter -> Virtualenv Environment -> New`.
   - Choose Python 3.10–3.12 (recommended for TensorFlow 2.16+).
3. **Install dependencies**. Open a terminal in PyCharm (bottom toolbar) and
   run:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`. Upload a
   chest X-ray PNG/JPG and view the prediction.

   Alternatively, add a PyCharm Run Configuration:
   - Run/Debug Configurations -> `+` -> `Python`
   - Script path: point at your Python's `streamlit` module, or simpler —
     use a **Shell Script** / **Terminal** configuration running
     `streamlit run app.py`.

5. **CLI inference** (no browser needed):
   ```bash
   python predict.py path/to/xray.png
   ```

## Notes on the model

- Input: grayscale image resized to **224×224**, pixel values scaled to
  **[0, 1]**.
- Architecture: 4 × (Conv2D → BatchNorm → MaxPool) blocks → Global Average
  Pooling → Dropout → Dense(128, relu) → Dropout → Dense(1, sigmoid).
- Output: a single probability of "Tuberculosis" (threshold 0.5 by default,
  adjustable in the app sidebar).
- Saved in the native Keras 3 format (`.keras`), created with
  `keras_version 3.13.2`. `tensorflow>=2.16` bundles a compatible Keras 3, so
  `tf.keras.models.load_model(...)` works directly — no separate `keras`
  package install needed.

## Retraining / reproducing the notebook locally

The full training pipeline (EDA, augmentation, training, evaluation) lives
in `pulmonary-chest-x-ray.ipynb`. To rerun it outside Kaggle:

1. Download the dataset from Kaggle
   (`kmader/pulmonary-chest-xray-abnormalities`) and place it locally.
2. Change the `INPUT_ROOT` path in the notebook (originally
   `/kaggle/input`) to point at your local copy.
3. Install the extra training-only dependencies already listed in
   `requirements.txt` (`scikit-learn`, `pandas`, `matplotlib`, `seaborn`).
4. Open the notebook in PyCharm's Jupyter support (or `jupyter lab`) and run
   all cells.

## Dataset attribution

- Jaeger S. et al., "Two public chest X-ray datasets for computer-aided
  screening of pulmonary diseases." *Quant Imaging Med Surg.* 2014.
- Jaeger S. et al., "Automatic tuberculosis screening using chest
  radiographs." *IEEE Trans Med Imaging.* 2014.
- Kaggle dataset:
  https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities
