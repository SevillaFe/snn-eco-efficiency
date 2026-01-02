# Eco-efficiency Analysis and Regression in Data Conversion for Spiking Neural Network Training

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official repository for the paper: **"Eco-efficiency analysis and regression in data conversion for Spiking Neural Network training"**

## Abstract

This study introduces a methodology to measure CO₂ emissions and energy consumption during both dataset encoding and training of Spiking Neural Networks (SNNs) in regression tasks. Using rate encoding with varied hyperparameters, we compare the original PilotNet CNN as a baseline with its SNN adaptation on the Udacity and Sully Chen driving datasets. The results demonstrate that temporal depth (S) is the dominant driver of environmental footprint, while conventional CPU/GPU platforms fail to leverage spike-driven temporal sparsity, indicating that true eco-efficiency gains will likely require neuromorphic hardware.

## Key Findings

- **Temporal depth (S)** is the primary driver of environmental cost in SNNs
- **Gain parameter (G)** has negligible impact on both footprint and performance
- CNN PilotNet outperforms SNNs on **Udacity dataset** (lower MSE: 0.0462 vs 0.0666, lower emissions: 0.31g vs 4.55g CO₂)
- SNN marginally surpasses CNN on **Sully Chen dataset** (MSE: 0.0274 vs 0.031) but at **~12.8× higher emissions** (18.83g vs 1.47g CO₂)
- Conventional hardware does not exploit spike-based temporal sparsity, limiting SNN eco-efficiency gains

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (optional)
│
├── data/
│   ├── README.md                      # Dataset download instructions
│   └── .gitkeep
│
├── src/
│   ├── CNN/
│   │   ├── PilotNet_PaloAlto_Dataset.py     # Udacity dataset
│   │   └── PilotNet_Udacity_Dataset.py      # Sully Chen dataset
│   │
│   ├── ETL/
│   │   ├── Encode_SNN_PaloAlto_Dataset.py     # ETL encoding for Udacity
│   │   └── Encode_SNN_Udacity_Dataset.py      # ETL encoding for Sully Chen
│   │
│   ├── SNN/
│   │   ├── SNN_LaskNet_PaloAlto_Dataset.py           # SNN LaskNet and h5 file for Sully Chen
│   │   ├── SNN_LaskNet_Udacity_Dataset.py            # SNN PilotNet and h5 file for Udacity
│   │   ├── SNN_PilotNet_PaloAlto_Dataset.py          # SNN LaskNet and h5 file for Sully Chen
│   │   └── SNN_PilotNet_Udacity_Dataset.py           # SNN PilotNet and h5 file for Udacity
│
├── results/
│   ├── figures/                       # Generated plots
│   ├── tables/                        # Result tables (CSV)
│   └── emissions/                     # CodeCarbon logs
│
├── paper/
│   ├── manuscript.tex                 # LaTeX source
│   ├── mybibfile.bib                  # Bibliography
│   └── figures/                       # Paper figures

```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (optional, for GPU training)
- 16GB RAM minimum
- ~50GB disk space for datasets and encoded versions

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/snn-eco-efficiency.git
cd snn-eco-efficiency
```

2. **Create virtual environment:**
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate snn-eco

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Download datasets:**
```bash
cd data
# Follow instructions in data/README.md
```

## Reproducing Paper Results

To reproduce the exact results from the paper:

```bash
# 1. Encode datasets with all configurations

# 2. Train all models

# 3. Generate plots and tables

```

## Results Summary

### Udacity Dataset

| Configuration | Val MSE | CO₂ (g) | Energy (Wh) |
|--------------|---------|---------|-------------|
| CNN Baseline | 0.0462  | 0.31    | 0.85        |
| SNN (S=5, G=0.5) | 0.0953  | 0.96    | 2.62        |
| SNN (S=25, G=1.0) | 0.0666  | 4.55    | 12.44       |

### Sully Chen Dataset

| Configuration | Val MSE | CO₂ (g) | Energy (Wh) |
|--------------|---------|---------|-------------|
| CNN Baseline | 0.031   | 1.47    | 4.03        |
| SNN (S=5, G=0.5) | 0.167   | 3.91    | 10.70       |
| SNN (S=25, G=1.0) | 0.0274  | 18.83   | 51.52       |

## Experimental Configuration

### Hardware
- **Platform:** Apple M1 Pro
- **CPU/GPU:** 16-core neural engine
- **RAM:** 16GB unified memory
- **Framework:** PyTorch Lightning 2.0+

### Hyperparameters

**Rate Encoding:**
- Temporal depth (S): [5, 15, 25]
- Gain (G): [0.5, 1.0]
- Image size: 66×200 pixels
- Grayscale conversion

**Training:**
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 64 (Udacity), 128 (Sully Chen)
- Epochs: 40
- Loss: MSE
- Precision: 16-bit mixed

**SNN Configuration:**
- Neuron model: Leaky Integrate-and-Fire (LIF)
- Beta: 0.9 (learnable)
- Threshold: 0.5
- Surrogate gradient: Fast sigmoid (slope=25)

## Dependencies

Core dependencies:
- `torch>=2.0.0`
- `pytorch-lightning>=2.0.0`
- `snntorch>=0.7.0`
- `h5py>=3.8.0`
- `codecarbon>=2.3.0`
- `pandas>=1.5.0`
- `numpy>=1.24.0`
- `Pillow>=9.5.0`
- `torchvision>=0.15.0`

See `requirements.txt` for complete list.

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{sevilla2026ecoefficiency,
  title={Eco-efficiency analysis and regression in data conversion for Spiking Neural Network training},
  author={Sevilla Mart{\'i}nez, Fernando and Casas-Roma, Jordi and Subirats, Laia and Parada, Ra{\'u}l},
  journal={Ecological Informatics},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

- **Project Link:** https://github.com/SevillaFe/snn-eco-efficiency

## Acknowledgments

- Universitat Oberta de Catalunya (UOC) - e-Health Center
- Volkswagen AG
- Universitat Autònoma de Barcelona - Computer Vision Center
- Centre Tecnològic de Telecomunicacions de Catalunya (CTTC/CERCA)
- Udacity for the driving simulator
- Sully Chen for the driving dataset

## Related Work

- [snnTorch Documentation](https://snntorch.readthedocs.io/)
- [CodeCarbon: Track emissions from compute](https://codecarbon.io/)
- [NVIDIA PilotNet Paper](https://arxiv.org/abs/1604.07316)

---

**Note:** This is research code. While we strive for reproducibility, results may vary slightly due to hardware differences, random initialization, and framework versions. For questions about reproducibility, please open an issue.
