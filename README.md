# MOGCN: Multi-Omics Graph Convolutional Network

**Integrated data analysis with graph machine learning using multi-omics and clinical datasets**

MOGCN is a deep learning framework that clusters patients based on their multi-omics data and clinical information using unsupervised graph autoencoders. This implementation enables comprehensive cancer subtype analysis by integrating multiple data modalities through graph neural networks.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

MOGCN provides a comprehensive pipeline for multi-omics cancer subtype analysis that:

- **Integrates multiple data modalities**: RNA-seq, DNA methylation, copy number variation, microRNA, protein expression, and clinical data
- **Uses advanced graph neural networks**: Implements Graph Autoencoders (GAE), Variational Graph Autoencoders (VGAE), and other architectures
- **Employs Similarity Network Fusion (SNF)**: Creates patient similarity graphs from multi-omics data
- **Enables unsupervised patient clustering**: Discovers cancer subtypes through latent representations
- **Supports survival analysis**: Provides clinical relevance assessment of discovered subtypes

### Key Features

- 🧬 **Multi-omics integration** with deep autoencoders
- 📊 **Patient similarity graphs** using SNF methodology  
- 🤖 **Graph neural network models** (GAE, VGAE, GraphSAGE, GAT)
- 🎯 **Unsupervised clustering** with UMAP and multiple clustering algorithms
- 📈 **Survival analysis** for clinical validation
- 🔬 **TCGA data compatibility** with XENA browser integration
## Methodology

MOGCN follows a four-stage pipeline for multi-omics analysis:

### 1. Multi-Modal Autoencoder Preprocessing
- Each omics modality is processed through dedicated encoders
- All encoders feed into a shared latent space for integrated representation
- Clinical features are appended to the latent representation
- **Requirement**: Each omics file must have Sample_ID as the first column

### 2. Patient Similarity Graph Generation  
- Similarity Network Fusion (SNF) creates patient similarity matrices
- Patients are connected by edges based on distance thresholds
- Feature and adjacency matrices are stored as PyTorch Geometric Data objects
- Multiple similarity metrics supported (cosine, euclidean, etc.)

### 3. Graph Autoencoder Training
- Graph Convolutional Networks integrate both feature and graph structure information
- Various architectures available: GAE, VGAE, GraphSAGE, GAT, and linear models
- Loss function: Mean squared error between original and reconstructed adjacency matrices
- Produces patient latent representations for downstream analysis

### 4. Clustering and Clinical Analysis
- Dimensionality reduction using UMAP or t-SNE
- Clustering with Agglomerative Clustering or DBSCAN
- Survival analysis using Kaplan-Meier estimators
- Clinical significance testing for discovered patient subtypes

## Data Requirements and XENA Browser Integration

This framework is designed to work seamlessly with data from the [UCSC Xena Browser](https://xenabrowser.net/), which provides access to TCGA and other large-scale cancer genomics datasets.

### Downloading Data from XENA Browser

1. **Visit XENA Browser**: Go to https://xenabrowser.net/
2. **Select Dataset**: Choose your cancer type (e.g., TCGA-LUAD, TCGA-BRCA)
3. **Download Multi-Omics Data**: 
   - Gene Expression (RNA-seq)
   - DNA Methylation (450K/EPIC arrays)  
   - Copy Number Variation
   - microRNA Expression
   - Protein Expression (RPPA)
   - Clinical/Phenotype Data

### Expected Data Format

All input files should be CSV format with the following structure:
```
Sample_ID,Feature_1,Feature_2,...,Feature_N
TCGA-XX-XXXX-01A,value1,value2,...,valueN
TCGA-XX-XXXX-01A,value1,value2,...,valueN
```

**Important Notes:**
- First column must be named `Sample_ID` 
- All values must be positive (preprocessing scripts handle this conversion)
- Missing values should be imputed or samples/features removed
- Sample IDs should follow TCGA naming convention (e.g., `TCGA-XX-XXXX-01A`)

### Data Preprocessing

Use the provided Jupyter notebooks in `Preprocessing Input Files/` to:

- **`preprocess_clinical.ipynb`**: Process XENA clinical and survival data
- **`preprocess_RNAseq.ipynb`**: Handle gene expression data  
- **`preprocess_methylation.ipynb`**: Process DNA methylation arrays
- **`preprocess_cnv.ipynb`**: Handle copy number variation data
- **`preprocess_miRNA.ipynb`**: Process microRNA expression
- **`preprocess_protein_array.ipynb`**: Handle protein expression (RPPA) data

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) 
- Conda environment manager (recommended)

### Environment Setup

1. **Create Conda Environment**
```bash
conda create -n mogcn python=3.8
conda activate mogcn
```

2. **Install PyTorch** (check [PyTorch website](https://pytorch.org/get-started/locally/) for latest versions)
```bash
# For CUDA 11.6
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

3. **Install PyTorch Geometric**
```bash
pip install torch-geometric
```

4. **Install Additional Dependencies**
```bash
pip install -r requirements.txt
```

5. **Install Weights & Biases** (for experiment tracking)
```bash
pip install wandb
wandb login  # Follow prompts to authenticate
```

### Verify Installation

```python
import torch
import torch_geometric
print(f"PyTorch version: {torch.__version__}")
print(f"PyG version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```


## Quick Start

### 1. Prepare Your Data

**Option A: Use XENA Browser Data**
1. Download multi-omics data from [UCSC Xena Browser](https://xenabrowser.net/)
2. Use preprocessing notebooks in `Preprocessing Input Files/` to format data
3. Ensure all files have `Sample_ID` as the first column

**Option B: Use Your Own Data**  
Ensure your CSV files follow the required format with positive values only.

### 2. Configure Data Paths

Edit the file paths in `train.py` or create a configuration file:

```python
# Example configuration in train.py
config.paths_omics = [
    'data/rnaseq_data.csv',
    'data/methylation_data.csv', 
    'data/cnv_data.csv',
    'data/mirna_data.csv'
]
config.path_overview = 'data/clinical_data.csv'
```

### 3. Start Experiment Tracking

```bash
# Start wandb server (optional, for local tracking)
wandb server start

# Or use wandb cloud (login required)
wandb login
```

### 4. Run the Pipeline

```bash
python train.py
```

### 5. Key Parameters to Customize

```bash
# Basic usage
python train.py --data_name LUAD --epochs 200 --device gpu

# Advanced configuration  
python train.py \
    --data_name LUAD \
    --epochs 200 \
    --lr 0.001 \
    --K 20 \
    --mu 0.6 \
    --metric sqeuclidean \
    --device gpu
```

### Parameter Guide

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_name` | Dataset identifier | Required |
| `--epochs` | Training epochs | 200 |
| `--lr` | Learning rate | 0.001 |
| `--K` | SNF neighbors | 20 |
| `--mu` | SNF normalization | 0.6 |
| `--metric` | Distance metric | sqeuclidean |
| `--device` | cpu/gpu | gpu |

## Example Workflow

Here's a complete example using TCGA lung cancer data:

```bash
# 1. Download TCGA-LUAD data from XENA Browser
# 2. Process using preprocessing notebooks
# 3. Configure and run

python train.py \
    --data_name LUAD \
    --paths_omics data/luad_rnaseq.csv data/luad_methylation.csv \
    --path_overview data/luad_clinical.csv \
    --epochs 200 \
    --lr 0.001 \
    --device gpu
```

## Output and Results

MOGCN generates several outputs:

- **Latent representations**: Patient embeddings for downstream analysis
- **Clustering results**: Patient subtype assignments  
- **Visualizations**: UMAP plots, survival curves, heatmaps
- **Performance metrics**: Silhouette scores, clustering quality measures
- **Clinical analysis**: Survival statistics, biomarker enrichment

Results are automatically logged to Weights & Biases for experiment tracking and comparison.

## Troubleshooting

### Common Issues

**GPU Memory Errors**
- Reduce batch size or use CPU mode
- Try smaller datasets for initial testing

**Data Format Errors** 
- Ensure Sample_ID is the first column
- Verify all values are positive numbers
- Check for missing values

**Installation Issues**
- Ensure CUDA versions match between PyTorch and system
- Use conda for better dependency management  
- Check PyTorch Geometric compatibility

### Getting Help

- Check the [Issues](https://github.com/ferdinand-popp/MOGCN/issues) page
- Review preprocessing notebooks for data format examples
- Ensure all dependencies are properly installed

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest improvements.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Ferdinand Popp - ferdinand.popp@proton.me

**Project Link**: [https://github.com/ferdinand-popp/MOGCN](https://github.com/ferdinand-popp/MOGCN)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements and Citations

This project builds upon several important works in multi-omics analysis and graph neural networks:

### Primary References

1. **Simple and Effective Graph Autoencoders with One-Hop Linear Models**  
   Salha, G. E., Hennequin, R., Tran, V. A., & Vazirgiannis, M. (2020)  
   *ArXiv preprint arXiv:2001.07614*  
   [Paper](https://arxiv.org/pdf/2001.07614v1.pdf)

2. **A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis**  
   Li, Y., Huang, H., Zhang, Y., Qian, W., & Deng, L. (2022)  
   *Frontiers in Genetics, 13*  
   [Paper](https://www.frontiersin.org/articles/10.3389/fgene.2022.806842/pdf)

### Key Methodologies

- **Similarity Network Fusion (SNF)**: Wang, B., et al. (2014). *Nature Methods, 11*(3), 333-337
- **PyTorch Geometric**: Fey, M., & Lenssen, J. E. (2019). *ArXiv preprint arXiv:1903.02428*
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). *ArXiv preprint arXiv:1802.03426*

### Data Sources

- **The Cancer Genome Atlas (TCGA)**: [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)
- **UCSC Xena Browser**: Goldman, M. J., et al. (2020). *Nature Biotechnology, 38*(6), 675-678  
  [https://xenabrowser.net/](https://xenabrowser.net/)

### Software and Libraries

- **PyTorch**: Paszke, A., et al. (2019). *Advances in Neural Information Processing Systems, 32*
- **Scikit-learn**: Pedregosa, F., et al. (2011). *Journal of Machine Learning Research, 12*, 2825-2830  
- **Pandas**: McKinney, W. (2010). *Proceedings of the 9th Python in Science Conference*
- **Weights & Biases**: [https://wandb.ai/](https://wandb.ai/)


