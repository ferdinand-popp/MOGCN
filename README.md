<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->
# WIP


# MOGCN: Integrated data analysis with graph machine learning using multi-omics and clinical datasets
Cluster patients based on their multiomics data and clinical utilizing unsupervised graph autoencoders. Adapted from the [Simple and Effective Graph Autoencoders with One-Hop Linear Models](https://arxiv.org/pdf/2001.07614v1.pdf)(Salha et al., 2020) and [A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis](https://www.frontiersin.org/articles/10.3389/fgene.2022.806842/pdf)(Li et al., 2022) in PyTorch Geometric.

## About the project
Project based on pytorch-geometric. It uses clinical EHR, and multi omics data from the TCGA Study [TCGA Study](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). The files are preprocessed to positive values.
### 1. Preprocess multi-omics data with a multi-modal autoencoder
Every omics file need the ID as the first column and each encoder feeds to a shared latent space, which integrates the representations of all omics for one patient.
Afterwards the latent representation is extracted and the clinical input features are appened.
### 2. Generate a patient similarity graph. 
Patient nodes (consisting of unprocessed omics and clinical data) are used to generate a similarity matrix with the similarity network fusion (SNF). Based on this matrix patient nodes have an edge connecting them if their distance is above a set threshold and no edge if it is below. The feature matrix and the adjacency matrix are stored in a PyTorch Data Object.
### 3. Train graph autoencoders 
GAE are graph convolutional nets that integrate feature and adjacency information. The resulting latent represenation is decoded to reconstruct the adjacency 
information and the loss is the mean squared error between the original matrix and the reconstructed one.
Various architectures from the pytorch geometric project are included and they all result in a latent representation after training. Mainly using **simple linear AE**, **GAE**, **VGAE**, **variational simple linear AE**, **GraphSAGE**, **GAT**, etc..
### 4. Clustering analysis for the latent represenation of the patients 
The latent represenation can the be projected via an dimensionality reduction (UMAP) and clustered (Agglomerative Clustering and DBSCAN). An survival analysis is performed on the clustered patients afterwards.

## Getting started
For GPU usage please check CUDA (min version 10.1) distributions in dependencies and in the requirements in the following links.
Conda environment preferred:
follow installation steps for pytorch under (min version 1.4.0): [Pytorch Installation](https://pytorch.org/get-started/locally/)

follow installation steps for pytorch geometric under (min version 1.6): [PyG Docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [PyG Installation](https://github.com/rusty1s/pytorch_geometric/blob/master/README.md#installation)

follow installation steps for Weights and Biases tracking under : [WandB - Quickstart](https://docs.wandb.ai/quickstart)

Remaining required packages under [Requirements](requirements.txt)


## Executing 
To run the pipeline, insert the file urls into train.py and check the parameters how the pipeline should be run (supervised or unsupervised). Afterwards spin up wandb: 'wandb server start' in terminal. Then you can run the main file:  
```python
train.py
```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Ferdinand Popp - ferdinand.popp@proton.me

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements WIP

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
