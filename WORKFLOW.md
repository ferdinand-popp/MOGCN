# MOGCN Workflow Diagram

```
                           MOGCN WORKFLOW
                    ================================

Step 1: Data Preparation (XENA Browser)
┌─────────────────────────────────────────────────────────────┐
│  UCSC XENA BROWSER (https://xenabrowser.net/)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ RNA-seq     │  │ Methylation │  │ Copy Number │  ...   │
│  │ Expression  │  │ 450K/EPIC   │  │ Variation   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │ microRNA    │  │ Clinical &  │                         │
│  │ Expression  │  │ Survival    │                         │
│  └─────────────┘  └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
                   Use preprocessing notebooks
                            ↓
Step 2: Multi-Modal Autoencoder Integration
┌─────────────────────────────────────────────────────────────┐
│  MULTI-OMICS DATA INTEGRATION                               │
│                                                             │
│  RNA-seq ──┐                                               │
│  Methyl. ──┤→ [Encoder 1] ──┐                             │
│  CNV    ──┐                 │                             │
│  miRNA  ──┤→ [Encoder 2] ──┐│→ [Shared Latent] → Patient  │
│  Protein──┘                 ││     Space          Embeddings│
│  Clinical──→ [Features] ────┘│                             │
│                              │                             │
│            [Decoder] ←───────┘                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
Step 3: Patient Similarity Graph Construction  
┌─────────────────────────────────────────────────────────────┐
│  SIMILARITY NETWORK FUSION (SNF)                           │
│                                                             │
│  Patient 1 ○────────○ Patient 2                           │
│           /│\      /│\                                     │
│          / │ \    / │ \                                    │
│         ○  │  ○  ○  │  ○                                   │
│      P4    │   P3    │   P5                               │
│            ○─────────○                                     │
│          Patient 6                                         │
│                                                             │
│  Similarity metrics: Cosine, Euclidean, etc.              │
│  Threshold-based edge connections                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
Step 4: Graph Neural Network Training
┌─────────────────────────────────────────────────────────────┐
│  GRAPH AUTOENCODERS                                         │
│                                                             │
│  Input: [Patient Graph] + [Features]                       │
│              ↓                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Graph Encoder (GAE/VGAE/GraphSAGE/GAT)                 ││
│  │   • Node features + Graph structure                     ││
│  │   • Message passing between patients                    ││
│  │   • Generate latent representations                     ││
│  └─────────────────────────────────────────────────────────┘│
│              ↓                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Graph Decoder                                           ││
│  │   • Reconstruct adjacency matrix                       ││
│  │   • MSE loss for reconstruction                        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                            ↓
Step 5: Patient Clustering & Analysis
┌─────────────────────────────────────────────────────────────┐
│  CLUSTERING & CLINICAL VALIDATION                           │
│                                                             │
│  Latent Representations                                     │
│         ↓                                                   │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Dimensionality  │    │ Clustering      │                │
│  │ Reduction       │    │ • Agglomerative │                │
│  │ • UMAP          │ →  │ • DBSCAN        │                │
│  │ • t-SNE         │    │ • K-means       │                │
│  └─────────────────┘    └─────────────────┘                │
│                                   ↓                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Clinical Analysis                                       ││
│  │ • Kaplan-Meier survival curves                         ││
│  │ • Statistical significance testing                      ││
│  │ • Biomarker enrichment analysis                        ││
│  │ • Subtype characterization                             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

Final Output: Cancer Patient Subtypes with Clinical Relevance
```

## Key Components:

### Input Data Format (from XENA):
```
Sample_ID,Gene_1,Gene_2,...,Gene_N
TCGA-XX-XXXX-01A,5.23,3.45,...,2.67
TCGA-XX-XXXX-01A,4.12,5.78,...,3.21
```

### Neural Network Architecture:
```
Multi-Modal Autoencoder → SNF Graph → Graph Autoencoder → Clustering
     ↓                      ↓              ↓               ↓
Patient embeddings → Patient graph → Latent space → Cancer subtypes
```

### Key Algorithms:
- **Similarity Network Fusion (SNF)**: Fuses multiple omics similarity matrices
- **Graph Autoencoders**: Learn patient representations from graph structure
- **UMAP/t-SNE**: Dimensionality reduction for visualization
- **Survival Analysis**: Kaplan-Meier estimation for clinical validation

This workflow transforms raw multi-omics data into clinically meaningful patient subtypes for precision medicine applications.