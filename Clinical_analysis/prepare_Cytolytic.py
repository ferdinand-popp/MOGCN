import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import plotly.io as pio
from clustering_analysis import plot_violin, add_p_value_annotation
from clinical_selectivity import chi_test, ensg_to_gene_name

pio.templates.default = "simple_white"
# run = wandb.init(project=f"Cluster_analysis", notes='setup')
# Load the data
# df_overview = pd.read_csv(r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUNG\TCGA_LUNG_overview_table.csv", index_col=1)
#
# df_merge = df_labels.merge(df_overview, on='Sample_ID', how='left')
# print('Patient Samples: ' + str(len(df_merge.index)))

path = r'D:\FPOPP\MoGCN\result\galant_sweep_14\labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

rna_path = r'Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\RNAseq\LUAD_RNA_seq.csv'
RNAseq = pd.read_csv(rna_path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID')
RNAseq.set_index('Sample_ID', inplace=True)
RNAseq.columns = ensg_to_gene_name(RNAseq.columns.to_list())
RNAseq.reset_index(inplace=True)
df_cytolytic = RNAseq[['Sample_ID', 'GZMA', 'PRF1']]
df_cytolytic['cyto_activity'] = RNAseq[['GZMA', 'PRF1']].apply(lambda x: x.mean(), axis=1)

df_merge = df_labels.merge(df_cytolytic, on='Sample_ID', how='left')
df_merge.sort_values('Labels', inplace=True)
print('Patient Samples: ' + str(len(df_merge.index)))

fig = plot_violin(df_merge, 'Labels', 'cyto_activity', yaxis="Mean expression of GZMA and PRF1")
fig = add_p_value_annotation(fig, [[0,1], [1,2], [0,2]])
fig.write_image(os.path.join(folder, f'Plot_cyto_activity.pdf'))

#save cytolytic to overview
df_overview = pd.read_csv(r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUNG\TCGA_LUNG_overview_table.csv", index_col=1)
#
df_subset = df_cytolytic[['Sample_ID', 'cyto_activity']]
df_overview_updated = df_overview.merge(df_subset, on='Sample_ID', how='left')
df_overview_updated.to_csv(r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUNG\TCGA_LUNG_overview_table.csv")
