import pandas as pd
import os
from clinical_selectivity import ensg_to_gene_name

path = r'D:\FPOPP\MoGCN\result\galant_sweep_14\labels.csv'
folder = os.path.dirname(path)
folder = os.path.join(folder, 'GSEA')

df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

rna_path = r'Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\RNAseq\TCGA_LUAD_RNAseq_log10_TPM.txt'  # r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\RNAseq\LUAD_RNA_seq_36000_unscaled.csv"
RNAseq = pd.read_csv(rna_path, sep='\t', index_col='Name')
RNAseq.columns = [x + '-01A' for x in RNAseq.columns]
RNAseq = RNAseq.T
RNAseq.reset_index(inplace=True)
RNAseq = RNAseq.rename(columns={"Name": "Sample_ID"})
RNAseq = RNAseq.rename(columns={"index": "Sample_ID"})

# RNAseq = pd.read_csv(rna_path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID')
# RNAseq.set_index('Sample_ID', inplace=True)
# # replace ENSG with Gene name
# RNAseq.columns = ensg_to_gene_name(RNAseq.columns.to_list())
# # drop duplicates with mean
# RNAseq = RNAseq.T
# RNAseq.reset_index(inplace=True)
# RNAseq = RNAseq.groupby('index').mean().reset_index()
# RNAseq = RNAseq.set_index('index')
# RNAseq = RNAseq.T
# print(RNAseq.isnull().any().any())
# RNAseq.reset_index(inplace=True)

df_merge = df_labels.merge(RNAseq, on='Sample_ID', how='inner')
df_merge.sort_values('Labels', inplace=True)
print('Patient Samples: ' + str(len(df_merge.index)))

# for each label sum all cols
df_final = pd.DataFrame()
df_group = df_merge.groupby('Labels')
# Loop through each group and plot the scatter plot
for name, df in df_group:
    df.set_index('Sample_ID', inplace=True)
    df.drop('Labels', axis=1, inplace=True)
    df = df.sum(axis=0)
    df_final[name] = df

filepath = os.path.join(folder, f'GSEA_all_Kay.rnk')
df_final.to_csv(filepath, sep='\t', index=True, header=False)

filepath = os.path.join(folder, f'GSEA_I_Kay.rnk')
df_final[['I']].to_csv(filepath, sep='\t', index=True, header=False)

filepath = os.path.join(folder, f'GSEA_II_Kay.rnk')
df_final[['II']].to_csv(filepath, sep='\t', index=True, header=False)

filepath = os.path.join(folder, f'GSEA_III_Kay.rnk')
df_final[['III']].to_csv(filepath, sep='\t', index=True, header=False)
print('Finished')
