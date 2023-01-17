import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from clustering_analysis import plot_violin, add_p_value_annotation

# load labels
path = r'D:\FPOPP\MoGCN\result\galant_sweep_14\labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

# load genomic score
path = r"Z:\HiWi\Popp\Analysis_LUAD_clusters_20230103\Violin_plots\TCGA_purity_ploidy_GD_cff_clonality_20230103.txt"
df_scores = pd.read_csv(path, sep='\t')
df_scores['Sample_ID'] = df_scores['sample'].str[:16]

df_merge = df_labels.merge(df_scores, on='Sample_ID', how='left')
print('Patient Samples: ' + str(len(df_merge.index)))

scores = ['purity', 'ploidy', 'Genome doublings', 'Cancer DNA fraction', 'Subclonal genome fraction']
for score in scores:
    fig = plot_violin(df_merge, 'Labels', score, yaxis=f"{score} of sample")
    fig = add_p_value_annotation(fig, [[0,1], [1,2], [0,2]])
    fig.write_image(os.path.join(folder, f'Plot_{score.replace(" ", "_")}.png'))

# load leukocyte score
path = r"Z:\HiWi\Popp\Analysis_LUAD_clusters_20230103\Violin_plots\TCGA_Leukocyte_score_20230103.txt"
df_leuko = pd.read_csv(path, sep='\t')
df_leuko['Sample_ID'] = df_leuko['SampleID'].str[:16]

df_merge = df_labels.merge(df_leuko, on='Sample_ID', how='left')
print('Patient Samples: ' + str(len(df_merge.index)))

fig = plot_violin(df_merge, 'Labels', 'leukocyte_score', yaxis="Leukocyte Score for each sample")
fig = add_p_value_annotation(fig, [[0, 1], [1, 2], [0, 2]])
fig.write_image(os.path.join(folder, f'Plot_leukocyte_score.pdf'))
