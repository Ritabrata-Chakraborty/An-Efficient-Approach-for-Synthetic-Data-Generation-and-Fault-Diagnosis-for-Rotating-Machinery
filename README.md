# An Efficient Approach for Synthetic Data Generation and Fault Diagnosis for Rotating Machinery

[![Conference](https://img.shields.io/badge/Conference-PHM%202025-blue.svg)](https://www.phm2025.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1049/icp.2025.2364-blue.svg)](https://doi.org/10.1049/icp.2025.2364)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Commercial-orange.svg)](LICENSE)

## 📄 Paper Information

This repository contains the official implementation of the paper:

**"An Efficient Approach for Synthetic Data Generation and Fault Diagnosis for Rotating Machinery"**

- **Authors**: Ritabrata Chakraborty, Tauheed Mian, Pradeep Kundu
- **Conference**: 15th Prognostics and System Health Management Conference (PHM 2025)
- **Venue**: KU Leuven, Belgium
- **Date**: June 2025
- **Publication**: IET Conference Proceedings 2025(10):241-247
- **DOI**: [10.1049/icp.2025.2364](https://doi.org/10.1049/icp.2025.2364)

## 🔍 Abstract

Fault diagnosis in rotating machinery is often hindered by the scarcity of fault samples and the resulting class imbalance. Generative Adversarial Networks (GANs) are effective at dealing with this issue. In this work, an efficient Auxiliary Classifier Wasserstein GAN with Gradient Penalty (ACWGAN-GP)-based framework for synthetic data generation and fault classification has been implemented. This approach leverages a Temporal Convolutional Network (TCN) discriminator and a 1D-CNN generator. Both models incorporate positional embeddings, enabling a single trained network to generate diverse time-series representations that mimic the variability observed in real-world data. To ensure and evaluate the quality of the generated samples, statistical similarity evaluations were conducted using four popular methods, including Pearson Correlation Coefficient (PCC), Cosine Similarity (CS), Kullback-Leibler Divergence (KLDiv), and Maximum Mean Discrepancy (MMD). The generated data effectively augments the limited fault samples, mitigating class imbalance and enhancing the robustness of fault diagnosis. In the proposed framework, the discriminator plays a dual role - it guides the generator during adversarial training and functions independently as a fault classifier. The results of different bearing faults are validated using the CWRU bearing dataset. The obtained results demonstrate the robustness and effectiveness of the present approach.

## 🌟 Key Contributions

- **Novel AC-WGAN Architecture**: Auxiliary Classifier Wasserstein GAN with Temporal Convolutional Networks for high-quality synthetic vibration data generation
- **Comprehensive Fault Diagnosis**: Multi-class bearing fault classification with improved accuracy through data augmentation
- **Statistical Validation**: Rigorous evaluation of generated data quality using statistical metrics and frequency domain analysis
- **Industrial Applicability**: Demonstrated effectiveness on the widely-used CWRU bearing dataset with practical implications

## 🏗️ Project Structure

```
📁 Predictive Maintenance/
├── 📓 Final.ipynb                 # Complete implementation pipeline
├── 📋 README.md                   # This documentation
├── 📂 Models/                     # Trained model files
│   ├── 🧠 GAN_Models/            # Generator and Discriminator models
│   │   └── *.pth                 # WGAN models with different configurations
│   └── 🎯 Classifiers/           # Fault classification models
│       └── *.pth                 # Trained classifier models
├── 📊 Generated_Data/            # Synthetic vibration signals
│   └── *_generated_*.csv         # Generated samples for each fault class
├── 📈 Results/                   # Experimental results and metrics
│   ├── *_generation_results.csv  # Generation quality metrics
│   ├── *_training_time.npy       # Training performance metrics
│   └── discriminator_training_metrics_*.csv # Classification results
├── 🎨 Plots/                     # Visualizations and analysis plots
├── 📁 Datasets/                  # Original dataset files
│   └── CWRU/                     # Case Western Reserve University dataset
│       ├── DE/                   # Drive End data
│       └── FE/                   # Fan End data
```

## 🚀 Key Features

### 🎯 Advanced GAN Architecture
- **Auxiliary Classifier WGAN**: Combines generation quality with classification accuracy
- **Temporal Convolutional Networks**: Captures temporal dependencies in vibration signals
- **Gradient Penalty**: Ensures stable training and high-quality generation

### 📊 Comprehensive Evaluation
- **Statistical Metrics**: Mean, standard deviation, skewness, and kurtosis comparison
- **Frequency Domain Analysis**: FFT-based spectral characteristic preservation
- **Classification Performance**: Fault diagnosis accuracy assessment
- **Computational Efficiency**: Training time and resource utilization analysis

### 🔧 Industrial Applications
- **Multi-Severity Fault Detection**: Handles varying fault severities and types
- **Real-time Compatibility**: Efficient inference for industrial monitoring systems
- **Scalable Architecture**: Adaptable to different machinery types and fault patterns

## 📋 Requirements

### Dependencies
```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
scipy>=1.7.0
tqdm>=4.62.0
```

## 🎯 Dataset Information

The implementation uses the **Case Western Reserve University (CWRU) Bearing Dataset**:

### Fault Classes (10 total)
| Class | Description | Severity Levels |
|-------|-------------|-----------------|
| **N** | Normal bearing | Baseline |
| **BA** | Ball faults | 7, 14, 21 mils |
| **IR** | Inner race faults | 7, 14, 21 mils |
| **OR** | Outer race faults | 7, 14, 21 mils |

### Data Characteristics
- **Sampling Frequency**: 12 kHz and 48 kHz
- **Signal Length**: Variable (standardized to 1024 samples)
- **Sensor Locations**: Drive End (DE) and Fan End (FE)
- **Load Conditions**: 0-3 HP motor loads

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/rotating-machinery-fault-diagnosis.git
cd rotating-machinery-fault-diagnosis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete Pipeline
```python
# Open and run Final.ipynb in Jupyter Notebook
jupyter notebook Final.ipynb
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{chakraborty2025efficient,
  title={An Efficient Approach for Synthetic Data Generation and Fault Diagnosis for Rotating Machinery},
  author={Chakraborty, Ritabrata and Mian, Tauheed and Kundu, Pradeep},
  booktitle={IET Conference Proceedings},
  volume={2025},
  number={10},
  pages={241--247},
  year={2025},
  publisher={IET},
  doi={10.1049/icp.2025.2364}
}
```

## 📧 Contact

For questions or collaboration opportunities:

- **Ritabrata Chakraborty**: [ritabratabits@gmail.com]

## 📜 License

This project is licensed under a Commercial License - see the [LICENSE](LICENSE) file for details. 

**⚠️ Commercial License Notice:**
- This software is proprietary and requires a paid license for commercial use
- Academic and research use may be permitted under specific terms
- Contact the authors for licensing terms and pricing information
- Unauthorized distribution or commercial use is prohibited

## 🙏 Acknowledgments

- Case Western Reserve University for providing the bearing dataset
- PHM 2025 Conference organizers
- KU Leuven for hosting the conference
- Open-source community for PyTorch and related libraries

---

<p align="center">
  <strong>⭐ If you find this work useful, please consider starring the repository! ⭐</strong>
</p>
