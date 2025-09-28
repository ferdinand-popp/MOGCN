# MOGCN Configuration Example
# Copy this file to config.py and modify paths according to your data

# Dataset configuration
DATA_NAME = "LUAD"  # Your dataset identifier
DEVICE = "gpu"  # "gpu" or "cpu"

# Data paths - Update these paths to point to your XENA browser data
PATHS_OMICS = [
    "data/luad_rnaseq.csv",          # RNA-seq expression data
    "data/luad_methylation.csv",     # DNA methylation data  
    "data/luad_cnv.csv",             # Copy number variation
    "data/luad_mirna.csv"            # microRNA expression
]

# Clinical/overview data path
PATH_OVERVIEW = "data/luad_clinical.csv"

# Training parameters
EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# SNF (Similarity Network Fusion) parameters
K = 20              # Number of neighbors
MU = 0.6           # Normalization factor
METRIC = "sqeuclidean"  # Distance metric

# Model parameters
HIDDEN_DIM = 128    # Hidden layer dimensions
LATENT_DIM = 64     # Latent space dimensions

# Clustering parameters
N_CLUSTERS = 4      # Number of clusters (if known)
MIN_SAMPLES = 5     # For DBSCAN clustering

# Output paths
RESULT_DIR = "results/"
ADJACENCY_MATRIX = "results/Similarity_fused_matrix.csv"
FEATURE_DATA = "results/latent_data.csv"

# Experiment tracking
WANDB_PROJECT = f"MoGCN_{DATA_NAME}_analysis"
WANDB_NOTES = "Multi-omics cancer subtype analysis"

# Optional: Patient subset (set to None to use all patients)
PATIENT_SUBSET = None

# Optional: Clinical features to append
APPEND_CLINICAL_FEATURES = ["age", "gender", "stage"]