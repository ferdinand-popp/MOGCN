import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clinical_selectivity import ensg_to_gene_name

#load labels
path = r'D:\FPOPP\MoGCN\result\galant_sweep_14\labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1


#load rnaseq gene symbol
rna_path = r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\RNAseq\LUAD_RNA_seq_36000_unscaled.csv"
RNAseq = pd.read_csv(rna_path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID')
RNAseq.set_index('Sample_ID', inplace=True)
#replace ENSG with Gene name
RNAseq.columns = ensg_to_gene_name(RNAseq.columns.to_list())
#drop duplicates with mean
RNAseq = RNAseq.T
RNAseq.reset_index(inplace=True)
RNAseq = RNAseq.groupby('index').mean().reset_index()
RNAseq = RNAseq.set_index('index')
RNAseq = RNAseq.T
print(RNAseq.isnull().any().any())
RNAseq.reset_index(inplace=True)

# merge both
df_merge = df_labels.merge(RNAseq, on='Sample_ID', how='inner')
df_merge.sort_values('Labels', inplace=True)
print('Patient Samples: ' + str(len(df_merge.index)))

#load gene lists
path= r"Z:\HiWi\Popp\Analysis_LUAD_clusters_20230103\Heatmap_gene_expression\Interested_gene_lists_20230103.txt"
df_genesets = pd.read_csv(path, sep='\t')

df_overview = df_merge.merge(df_genesets)

sns.clustermap(df_heatmap, z_score=0, cmap="vlag", center=0) #, standard_scale=1) # ,z_score=0, cmap="vlag", center=0)
plt.title(f'Z score for Expression of MHC Genesets combined')
#plt.show()
plt.savefig(os.path.join(folder, 'Heatmaps', 'MHC_Genesets_combined.png'), dpi=300)

