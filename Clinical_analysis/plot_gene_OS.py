import pandas as pd
import os
import numpy as np
from clinical_selectivity import create_survival_plot, ensg_to_gene_name

# load labels
path = 'sweeps/labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

# Load the data
df_overview = pd.read_csv("data/TCGA_LUNG_overview_table.csv", index_col=1)
df_merge = df_labels.merge(df_overview, on='Sample_ID', how='left')
print('Patient Samples: ' + str(len(df_merge.index)))

# load rnaseq gene symbol
rna_path = "data/LUAD_RNA_seq_36000_unscaled.csv"
RNAseq = pd.read_csv(rna_path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID')
RNAseq.set_index('Sample_ID', inplace=True)
# replace ENSG with Gene name
RNAseq.columns = ensg_to_gene_name(RNAseq.columns.to_list())
# drop duplicates with mean
RNAseq = RNAseq.T
RNAseq.reset_index(inplace=True)
RNAseq = RNAseq.groupby('index').mean().reset_index()
RNAseq = RNAseq.set_index('index')
RNAseq = RNAseq.T
print(RNAseq.isnull().any().any())
RNAseq.reset_index(inplace=True)

# merge both
df_combined = df_merge.merge(RNAseq, on='Sample_ID', how='inner')
df_combined.sort_values('Labels', inplace=True)
print('Patient Samples: ' + str(len(df_combined.index)))

gene_list = ['BTLA', 'CD40LG', 'HLA-DQB2', 'HLA-DQB1', 'HLA-DQA1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DRB1']
print(set(gene_list).intersection(set(RNAseq.columns)))
df_genes = df_combined[gene_list]
df_genes[['Sample_ID', 'OS', 'OS.time']] = df_combined[['Sample_ID', 'OS', 'OS.time']]

# #
# #df_genes['scores_mean'] = df_genes[gene_list].mean(axis=1)
# for gene in gene_list:
#     #df_genes['high_low'] = np.where(df_genes[gene]>df_genes['mean'], 'high', 'low')
#     fig = create_survival_plot(df_genes, labels='high_low', time='OS.time', event='OS', years=5)
#     fig.show()
#     #fig.savefig(os.path.join(folder, 'clinical', f'Plot_OS_{gene}.pdf'))

for gene in gene_list:
    df_genes['high_low'] = np.where(df_genes[gene]>df_genes[gene].mean(), 'high', 'low')
    fig = create_survival_plot(df_genes, labels='high_low', time='OS.time', event='OS', years=5)
    fig.show()
    fig.savefig(os.path.join(folder, 'clinical', f'Plot_OS_{gene}.pdf'))

#plot groups
df_genes['MHC'] = df_genes[['HLA-DQB2', 'HLA-DQB1', 'HLA-DQA1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DRB1']].mean(axis=1)
df_genes['high_low'] = np.where(df_genes['MHC'] > df_genes['MHC'].mean(), 'high', 'low')
fig = create_survival_plot(df_genes, labels='high_low', time='OS.time', event='OS', years=5)
fig.show()
fig.savefig(os.path.join(folder, 'clinical', f'Plot_OS_MHC.pdf'))
#others
df_genes['Checkpoint'] = df_genes[['BTLA', 'CD40LG']].mean(axis=1)
df_genes['high_low'] = np.where(df_genes['Checkpoint'] > df_genes['Checkpoint'].mean(), 'high', 'low')
fig = create_survival_plot(df_genes, labels='high_low', time='OS.time', event='OS', years=5)
fig.show()
fig.savefig(os.path.join(folder, 'clinical', f'Plot_OS_Checkpoint.pdf'))
