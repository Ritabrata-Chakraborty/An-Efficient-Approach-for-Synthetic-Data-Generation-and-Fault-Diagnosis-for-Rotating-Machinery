# An Efficient Approach for Synthetic Data Generation and Fault Diagnosis for Rotating Machinery

[![Paper](https://img.shields.io/badge/Conference-PHM%202025-blue.svg)]([https://www.phm2025.org/](https://ieeexplore.ieee.org/document/11090096))
[![DOI](https://img.shields.io/badge/DOI-10.1049/icp.2025.2364-blue.svg)](https://doi.org/10.1049/icp.2025.2364)

## Paper Information

This repository contains the official implementation of the paper:

**"An Efficient Approach for Synthetic Data Generation and Fault Diagnosis for Rotating Machinery"**

- **Authors**: [Ritabrata Chakraborty](https://in.linkedin.com/in/ritabrata-chakraborty-a63268251/), Tauheed Mian, Pradeep Kundu
- **Conference**: 15th Prognostics and System Health Management Conference (PHM 2025)
- **Venue**: KU Leuven, Belgium
- **Date**: June 2025
- **Publication**: IET Conference Proceedings 2025(10):241-247
- **DOI**: [10.1049/icp.2025.2364](https://doi.org/10.1049/icp.2025.2364)

## Abstract

Fault diagnosis in rotating machinery is often hindered by the scarcity of fault samples and the resulting class imbalance. Generative Adversarial Networks (GANs) are effective at dealing with this issue. In this work, an efficient Auxiliary Classifier Wasserstein GAN with Gradient Penalty (ACWGAN-GP)-based framework for synthetic data generation and fault classification has been implemented. This approach leverages a Temporal Convolutional Network (TCN) discriminator and a 1D-CNN generator. Both models incorporate positional embeddings, enabling a single trained network to generate diverse time-series representations that mimic the variability observed in real-world data. To ensure and evaluate the quality of the generated samples, statistical similarity was evaluated using four popular methods: Pearson Correlation Coefficient (PCC), Cosine Similarity (CS), Kullback-Leibler Divergence (KLDiv), and Maximum Mean Discrepancy (MMD). The generated data effectively augment the limited number of fault samples, mitigating class imbalance and enhancing the robustness of fault diagnosis. In the proposed framework, the discriminator plays a dual role: it guides the generator during adversarial training and serves as an independent fault classifier. The results of different bearing faults are validated using the CWRU bearing dataset. The obtained results demonstrate the robustness and effectiveness of the present approach.

## Project Structure

```
Predictive Maintenance/
├── Final.ipynb                     
├── README.md                       
├── Models/                                  # Trained model files
│   ├── GAN_Models/                          # Generator and Discriminator models
│   │   └── *.pth                      
│   └── Classifiers/                         # Fault classification models
│       └── *.pth                  
├── Generated_Data/                          # Synthetic vibration signals
│   └── *_generated_*.csv         
├── Results/                                 # Experimental results and metrics
│   ├── *_generation_results.csv  
│   ├── *_training_time.npy      
│   └── discriminator_training_metrics_*.csv 
├── Plots/                                   # Visualizations and analysis plots
├── Datasets/                                # Original dataset files
│   └── CWRU/                     # Case Western Reserve University dataset
│       ├── DE/                   
│       └── FE/                   
```
## Requirements

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

## Dataset Information

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

## Quick Start

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

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{11090096,
  author={Chakraborty, Ritabrata and Mian, Tauheed and Kundu, Pradeep},
  booktitle={15th Prognostics and System Health Management Conference (PHM 2025)}, 
  title={An efficient approach for synthetic data generation and fault diagnosis for rotating machinery}, 
  year={2025},
  volume={2025},
  number={},
  pages={241-247},
  keywords={},
  doi={10.1049/icp.2025.2364}
}
```

---

<p align="center">
  <strong>If you find this work useful, please consider starring the repository!</strong>
</p>
