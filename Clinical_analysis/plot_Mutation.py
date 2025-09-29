import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import plotly.io as pio
from clustering_analysis import plot_violin
from clinical_selectivity import ensg_to_gene_name
from matplotlib_venn import venn3, venn3_circles


def plot_venn(df):
    fig = plt.Figure()
    set1 = set(df.iloc[:, 0].to_list())
    set2 = set(df.iloc[:, 1].to_list())
    set3 = set(df.iloc[:, 2].to_list())

    venn3([set1, set2, set3],
          set_labels=df.columns.to_list(),
          set_colors=('blue', 'yellow', 'green'),
          alpha = 0.3)

    return fig


pio.templates.default = "simple_white"
# run = wandb.init(project=f"Cluster_analysis", notes='setup')
# Load the data
# df_overview = pd.read_csv(r"data\TCGA_LUNG_overview_table.csv", index_col=1)
#
# df_merge = df_labels.merge(df_overview, on='Sample_ID', how='left')
# print('Patient Samples: ' + str(len(df_merge.index)))

# load labels
path = 'sweeps/labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

# # load omics
# path = r'data\TCGA_LUAD_mutation2.csv' #normal or missense
# df_mutation = pd.read_csv(path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID')
# df_mutation.set_index('Sample_ID', inplace=True)
#
# # loop through labels and grab top 20 mutations
# df_final = pd.DataFrame()
# df_group = df_labels.groupby('Labels')
# # Loop through each group and plot the scatter plot
# for name, df in df_group:
#     df_mut_label = df_mutation[df_mutation.index.isin(df['Sample_ID'].to_list())]
#     # sum all patients to get incidence for each gene
#     df_sum = df_mut_label.sum(axis=0)
#     df_sum = df_sum.sort_values(ascending=False)
#     #append top 20
#     mutation_list = df_sum[:20].index.to_list()
#     df_final[name] = mutation_list
#
# df_final
# df_final.to_csv(os.path.join(folder, f'top20_mutation.csv'))
#
# #give out the unique mutations
# a, b, c = set(df_final.iloc[:,0]), set(df_final.iloc[:,1]), set(df_final.iloc[:,2])
only_a = a - b - c
only_b = b - a - c
only_c = c - a - b
not_a = b.intersection(c) - a
not_b = a.intersection(c) - b
not_c = a.intersection(b) - c
print(only_a)
print(only_b)
print(only_c)
print(not_a)
print(not_b)
print(not_c)


url = 'data/TCGA-LUAD.mutect2_snv.tsv' # r"Z:\HiWi\Popp\TCGA_Breast_2022\TCGA-BRCA.mutect2_snv.tsv" # r'Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\Somatic_mutation\TCGA-LUAD.mutect2_snv.tsv'
df_LUAD = pd.read_csv(url, sep='\t', index_col=0)
df_LUAD = df_LUAD[['gene', 'effect']] # kick chr and vaf

#filter effects
include_effect = ['missense_variant'] # ['stop_gained', 'stop_lost', 'missense_variant', 'frameshift_variant']
df_mut_long = df_LUAD[df_LUAD.effect.isin(include_effect)]
df_mut_long.reset_index(inplace=True)

df_merge = df_mut_long.merge(df_labels, on='Sample_ID', how='left')
# loop through labels and grab top 20 mutations
df_group = df_merge.groupby('Labels')
# Loop through each group and plot the scatter plot
for name, df_sub in df_group:
    # if name == 'I':
    #     subset = a
    # elif name == 'II':
    #     subset = b
    # else:
    #     subset = c
    #
    # df_subset = df_sub[df_sub.gene.isin(subset)]
    # df_subset.drop('Labels', axis=1, inplace=True)
    df_subset = df_sub.drop('Labels', axis=1)
    df_subset = df_subset.drop_duplicates()
    counts = df_subset.groupby(['gene', 'effect']).size().reset_index(name='count')
    counts['occurence'] = (counts['count'] / len(df_sub.Sample_ID.unique()))*100
    counts.sort_values('occurence', inplace=True, ascending=False)
    counts = counts.iloc[:20,]
    color_map = {'frameshift_variant': '#AB63FA',
                 'missense_variant':  '#FFA15A',
                 'stop_gained': '#19D3F3',
                 'stop_lost': '#FF6692'}
    fig = px.bar(counts, x='gene', y='occurence', color='effect', title=name, color_discrete_map=color_map)
    fig.update_layout(yaxis=dict(range=[0, 65]), xaxis_title="Mutated genes", yaxis_title="Percentage of mutation occurence")
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(tickangle=270)
    fig.update(layout_showlegend=False)
    fig.show()
    fig.write_image(os.path.join(folder, 'mutations',f'top20_{name}.svg'))

a = set(counts.gene)
