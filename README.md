# Wavelet Interface Network (WIN) for Audio Deepfake Detection

![Python(Preferred)](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook%20Ready-20BEFF)

This repository contains the official implementation of the **Wavelet Interface Network (WIN)** for audio deepfake detection using wavelet-based feature mapping and transformer-style modeling.

The proposed model integrates signal preprocessing, learnable Sinc-based frontend, positional aggregation, and multi-wavelet attention for robust anti-spoofing.

The framework supports multiple analytic wavelet families, enabling systematic ablation and comparative analysis.

---

## 📌 Key Features

- End-to-end learning from raw waveform
- Pre-emphasis filtering
- Sinc-based convolutional frontend
- CNN feature extraction
- Positional encoding
- Multi-wavelet attention mechanism
- Transformer-style encoder
- Attention-based sequence pooling
- Support for multiple wavelet families
- EER and t-DCF evaluation
- FLOPs and parameter analysis

---

## 📁 Project Structure

```

WIN/
│
├── train.py              # Training script
├── test.py               # Testing / evaluation
├── model_info.py         # Parameter & FLOPs analysis (optional)
│
├── config.py             # Configuration
├── requirements.txt      # Dependencies
│
├── utils/
│   ├── device.py
│   └── metrics.py
│
├── data/
│   └── dataloader.py
│
├── models/
│   ├── preprocess.py
│   ├── frontend.py
│   ├── encoder.py
│   ├── WIN_classifier.py
│   └── WIN.py
│
├── tests/
│   └── test_forward.py
│
└── README.md

````

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Optional tools for model analysis:

```bash
pip install torchinfo fvcore
```

---

## 📊 Dataset Structure

Organize your dataset as:

```
dataset_root/
├── train/
│   ├── bonafide/
│   └── spoof/
├── dev/
│   ├── bonafide/
│   └── spoof/
└── test/
    ├── bonafide/
    └── spoof/
```

Update dataset paths in `config.py`.

---

## 🚀 Training

Run training:

```bash
python train.py
```

The best model is saved automatically based on validation EER.

---

## 🧪 Testing

Run evaluation:

```bash
python test.py
```

Outputs:

* Final EER
* Minimum t-DCF

---

## 🔍 Sanity Check

Verify forward pass and architecture:

```bash
python tests/test_forward.py
```

This performs a dummy inference to validate model consistency.

---

## 📐 Model Complexity

Check parameters and FLOPs:

```bash
python model_info.py
```

This reports:

* Trainable parameters
* Total parameters
* Model size
* MACs / FLOPs
* GFLOPs per second
* Layer-wise breakdown

---

## 🧠 Model Architecture

The overall processing pipeline is:

```
Waveform
   ↓
Pre-Emphasis
   ↓
Sinc + CNN Frontend
   ↓
Positional Encoding
   ↓
Multi-Wavelet Transformer
   ↓
Sequence Pooling
   ↓
Classifier
```

---

## 🌊 Supported Wavelet Families

The Wavelet-FAN attention module supports the following wavelet types:

| Config Name | Wavelet Family               |
| ----------- | ---------------------------- |
| bump        | Bump Wavelet                 |
| morlet      | Morlet Wavelet               |
| dog         | Derivative of Gaussian (DoG) |
| mexican     | Mexican Hat (Ricker)         |
| morse       | Generalized Morse            |

Wavelet type can be selected in `config.py`:

```python
WAVELET_TYPE = "bump"   # default
```

---

## 📈 Evaluation Metrics

The following metrics are used:

* Equal Error Rate (EER)
* Tandem Detection Cost Function (t-DCF)

Implemented in `utils/metrics.py`.

---

## 🔧 Configuration

All hyperparameters and experiment settings are defined in:

```
config.py
```

This includes:

* Dataset paths
* Training parameters
* Model dimensions
* Wavelet selection

Modify this file to conduct different experiments.

---

## 💾 Checkpoints

Trained models are saved at:

```
WIN.pth
```

Defined in `config.py` as `SAVE_PATH`.

---

## 📄 Citation

If you use this work in your research, please cite:

```
@article{win2026,
  title={Wavelet Interface Network for Audio Deepfake Detection},
  author={Shah, Arth J. and Pandey, Aniket and Patil, Hemant A.},
  journal={Journal/Conference},
  year={2026}
}
```

(Replace with the final publication details.)

---

## 📜 License

This project is intended for academic and research use only.

For commercial usage, please contact the authors.

---

## 🙏 Acknowledgements

* ASVspoof Challenge
* PyTorch
* torchaudio
* fvcore
* torchinfo

---

## 📬 Contact

Authors:
Arth J. Shah
Aniket Pandey
Hemant A. Patil

Email:
[202521004@dau.ac.in](mailto:202521004@dau.ac.in)
[202411001@dau.ac.in](mailto:202411001@dau.ac.in)

```
If you’d like, next I can help you prepare a **“Reproducibility Checklist” section** for top-tier conferences/journals.
```
