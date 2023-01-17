# FOR ALL
import pandas as pd
import wandb
from train import calculate_overlap
from sklearn.metrics import silhouette_score

from train import plot_embedding, clustering_points, plot_silhouette_comparison, survival_analysis, \
    clustering_points_agglo, projection
from clinical_selectivity import run_clinical_selectivity

wandb.init(project="MoGCN_LUNG_jDR")

data_name = 'LUNG'  # results20221006214008 # results20221224124726 # results20221227115340
methods = {'MCIA': r"D:\FPOPP\momix\results20221227232925\factors_mcia.csv",
           'MOFA2': r'D:\FPOPP\momix\results20221227232925\factors_mofa.csv',
           'intNMF': r"D:\FPOPP\momix\results20221227232925\factors_intNMF.csv"
           }

# grab clinical annotation
url_clinical = r'Z:\HiWi\Popp\TCGA_NSCLC_2022\LUNG\TCGA_LUNG_overview_table.csv'
clinical_df_raw = pd.read_csv(url_clinical, header=0, index_col=None).iloc[:, 1:]
clinical_df_raw.sort_values(by='Sample_ID', ascending=True, inplace=True)
clinical_df_raw.set_index('Sample_ID', inplace=True)
clinical_df_raw.dropna(subset=['disease_code'], inplace=True)

# iter methods
for method, url in methods.items():
    print(f"Analysing {method}...")
    # grab method result samples x 2 dims
    df = pd.read_csv(url)
    latent_data_2D = projection(df.iloc[:, 1:].values, dimensions=2, projection_type='UMAP')
    result_df = pd.concat([df.iloc[:, 0], latent_data_2D], axis=1)
    result_df.columns = ["Sample_ID", "1", "2"] if result_df.shape[1] == 3 else ["Sample_ID", "1", "2", "3"]
    result_df.Sample_ID = result_df.Sample_ID.apply(lambda x: x.replace(".", "-"))
    result_df.set_index('Sample_ID', inplace=True)

    # overlap of both jDR and clinical
    overlap = list(set(clinical_df_raw.index.tolist()) & set(result_df.index.tolist()))
    # subset both and sort
    result_df = result_df.loc[overlap].sort_index()
    result_df = result_df.reset_index(drop=True)
    clinical_df = clinical_df_raw.loc[overlap].sort_index().reset_index()

    # Plot clustering with LUAD LUSC
    plot_embedding(result_df, labels=clinical_df.disease_code, type=f'jDR_{method}_{data_name}',
                   title=f"jDR_{method}_{data_name}")

    score, minority_indices = calculate_overlap(result_df['1'].to_list(), result_df['2'].to_list(),
                                                clinical_df.disease_code)
    print('Adj_rand_Score_AE:' + str(score))
    print(
        'Inspect these patients as they dont group right:' + str(
            clinical_df.disease_code[minority_indices].index.to_list()))
    silhouette_avg = silhouette_score(result_df.values, clinical_df.disease_code.to_list())
    print('Silhoutte_Score_LUAD_LUSC_AE:' + str(silhouette_avg))
    wandb.log({f'Adj_rand_Score_AE_{method}': score, f'Silhoutte_Score_LUAD_LUSC_AE_{method}': silhouette_avg})

    if True:
        # Clustering and Plot of the projection into patient groups
        # labels, sil_score = clustering_points(result_df)
        # if labels is not None:
        #     plot_embedding(result_df, labels=labels, type=f'jDR_{method}_DBSCAN',
        #                    title=f"jDR_{method}_DBSCAN: Silhoutte Score{sil_score}")

        # Agglo
        labels, sil_score = clustering_points_agglo(result_df)
        plot_embedding(result_df, labels=labels, type=f'jDR_{method}_Agglomerative_Clustering',
                       title=f"jDR_{method}_Agglomerative_Clustering: Silhoutte Score{sil_score}")

        if 1 in labels.values:
            # df_clinical_selectivity = run_clinical_selectivity(labels, clinical_df)
            # wandb.log({f'{method}_clinical_selectivity': wandb.Table(dataframe=df_clinical_selectivity, columns=df_clinical_selectivity.columns.astype(str))})

            # Plot average groups silhouette score
            plot_silhouette_comparison(result_df, labels)

            survival_analysis(clinical_df[['OS', 'OS.time']], labels)
