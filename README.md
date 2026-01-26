# Wavelet Interface Network (WIN) for Audio Deepfake Detection

![Python(Preferred)](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook%20Ready-20BEFF)

This repository contains the official implementation of the **Wavelet Interface Network (WIN)** for audio deepfake detection using wavelet-based feature mapping and transformer-style modeling.

The proposed model integrates signal preprocessing, learnable Sinc-based frontend, positional aggregation, and wavelet-based attention for robust anti-spoofing.


---

## Project Structure
```
WIN/
│
├── train.py              # Training script
├── test.py               # Testing / evaluation
├── model_info.py   (optional, requires optional library installation)      # Parameter & FLOPs analysis 
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
```
---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

Optional tools:

```bash
pip install torchinfo fvcore
```

---

## Dataset Structure

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

Update paths in `config.py`.

---

## Training

Run training:

```bash
python train.py
```

The best model is saved automatically.

---

## Testing

Run evaluation:

```bash
python test.py
```

Outputs final EER and min-tDCF.

---

## Sanity Check

Verify forward pass:

```bash
python tests/test_forward.py
```

---

## Model Complexity

Check parameters and FLOPs:

```bash
python model_info.py
```

---

## Model Architecture

Pipeline:

```
Waveform
   ↓
Pre-Emphasis
   ↓
Sinc + CNN Frontend
   ↓
Positional Encoding
   ↓
Wavelet Transformer
   ↓
Sequence Pooling
   ↓
Classifier
```

---

## Evaluation Metrics

* Equal Error Rate (EER)

Implemented in `utils/metrics.py`.

---

## Configuration

All hyperparameters are in:

```
config.py
```

Modify for experiments.

---

## Checkpoints

Saved at:

```
WIN.pth
```

---

## Citation

If you use this work, please cite:

```
@article{win2026,
  title={Wavelet Interface Network for Audio Deepfake Detection},
  author={Author Names},
  journal={Journal/Conference},
  year={2026}
}
```

---

## License

For academic and research use only.

---

## Acknowledgements

* ASVspoof Challenge
* PyTorch
* torchaudio
* fvcore
* torchinfo

---

## Contact

Authors: Arth J. Shah, Aniket Pandey, Hemant A. Patil

Email: [202521004@dau.ac.in](mailto:202521004@dau.ac.in), [202411001@dau.ac.in](mailto:202411001@dau.ac.in)
