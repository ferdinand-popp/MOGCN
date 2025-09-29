# Data Preparation Guide for MOGCN

This guide explains how to prepare your data for use with MOGCN, specifically focusing on data downloaded from the UCSC Xena Browser.

## Step 1: Download Data from XENA Browser

1. Visit [UCSC Xena Browser](https://xenabrowser.net/)
2. Navigate to "Data Sets" 
3. Select your cancer type from GDC datasets (e.g., "GDC TCGA Lung Adenocarcinoma (LUAD)")
4. Download the following data types:

### Required Data Files:
- **Gene Expression**: RNA-seq data (e.g., "gene expression RNAseq - STAR FPKM")
- **Clinical Data**: Patient phenotype and survival information
- **DNA Methylation**: 450K or EPIC methylation arrays (optional)
- **Copy Number**: Gene level copy number data e.g. ASCAT3 (optional)  
- **microRNA**: miRNA expression data (optional)
- **Protein**: RPPA protein expression data (optional)
- **Somatic Mutation Calls**: SNPs and small InDels (e.g. WXS) (optional)


## Step 2: Data Format Requirements

All data files must be in CSV format with this structure:

```
Sample_ID,Feature_1,Feature_2,...,Feature_N
TCGA-05-4384-01A,5.234,2.145,...,8.456
TCGA-05-4390-01A,4.123,3.678,...,7.234
```

### Key Requirements:
- **First column**: Must be named `Sample_ID`
- **Sample IDs**: Should follow TCGA format (e.g., `TCGA-XX-XXXX-01A`)
- **Values**: All numeric values must be positive
- **Missing values**: Should be handled (imputed or removed)

## Step 3: Use Preprocessing Notebooks

Use the provided Jupyter notebooks in `Preprocessing_input_files/` to convert XENA data:

### For Clinical Data:
```bash
jupyter notebook "Preprocessing_input_files/preprocess_clinical.ipynb"
```
- Processes survival and phenotype data
- Handles sample ID formatting
- Creates overview table with clinical features

### For RNA-seq Data:
```bash
jupyter notebook "Preprocessing_input_files/preprocess_RNAseq.ipynb"
```
- Normalizes expression values
- Filters low-expression genes
- Ensures positive values

### For Other Omics:
- `preprocess_methylation.ipynb`: DNA methylation preprocessing
- `preprocess_cnv.ipynb`: Copy number variation
- `preprocess_miRNA.ipynb`: microRNA expression
- `preprocess_protein_array.ipynb`: Protein expression
- `preprocess_somatic_mutation.ipynb`: Somatic mutation data preprocessing


## Step 4: Data Validation

Before running MOGCN, verify your data:

```python
import pandas as pd

# Check data format
df = pd.read_csv('your_data.csv')
print(f"Shape: {df.shape}")
print(f"First column: {df.columns[0]}")
print(f"Sample ID example: {df.iloc[0, 0]}")

# Check for negative values
has_negatives = (df.iloc[:, 1:] < 0).any().any()
print(f"Has negative values: {has_negatives}")

# Check for missing values
missing_count = df.isnull().sum().sum()
print(f"Missing values: {missing_count}")
```

## Step 5: Update Configuration

Copy `config_example.py` to `config.py` and update the paths:

```python
# Update these paths in config.py
PATHS_OMICS = [
    "data/your_rnaseq_data.csv",
    "data/your_methylation_data.csv",
    # ... add other omics files
]

PATH_OVERVIEW = "data/your_clinical_data.csv"
```

## Example File Structure

After preparation, your data directory should look like:

```
data/
├── luad_clinical.csv
├── luad_rnaseq.csv
├── luad_methylation.csv
├── luad_cnv.csv
└── luad_mirna.csv
```

Each file should have the same Sample_IDs in the first column for proper integration.