# DynAttGraph2Seq
DynAttGraph2Seq is a novel framework to model complex dynamic transitions of an individual user's activities and the textual information of the posts over time in online health forums and learning how these correspond to the health stage development. 
To achieve this, we first formulate the transition of user activities as a dynamic attributed graph with multi-attributed nodes that evolves over time, then formalize the health stage inference task as a dynamic attributed graph to sequence learning problem. 
Our proposed model consists of a novel dynamic graph encoder along with a two-level sequential encoder to capture the semantic features from user posts and an interpretable sequence decoder that learns the mapping between a sequence of time-evolving user activity graphs as well as user posts to a sequence of target health stages. 
We go on to propose new dynamic graph regularization and dynamic graph hierarchical attention mechanisms to facilitate the necessary multi-level interpretability.
A comprehensive experimental analysis on health stage prediction tasks demonstrates both the effectiveness and the interpretability of the proposed models.

## Links
Paper: [Gao, Yuyang, Tanmoy Chowdhury, Lingfei Wu, and Liang Zhao. "Modeling Health Stage Development of Patients With Dynamic Attributed Graphs in Online Health Communities." IEEE Transactions on Knowledge and Data Engineering 35, no. 2 (2022): 1831-1843.](https://ieeexplore.ieee.org/document/9684988)

## Instructions:
1. Main Model:
Use [run_model.py](/main/run_model.py) to run the model and [configure.py](/main/configure.py) to configure the model.
2. SOTA Model: 
Use [run_nmt.py](/seq2seq_pytorch/run_nmt.py) to run the seq2seq model.

### Data: 
1. [Breast Cancer](https://cs.emory.edu/~lzhao41/pages/dataset_pages/social_media_online_health_forum_dataset.htm)
2. [Bladder Cancer](/data_processing/bladder_cancer)

## Citation
If you use this work, please cite the following paper.

"Gao, Yuyang, Tanmoy Chowdhury, Lingfei Wu, and Liang Zhao. "Modeling Health Stage Development of Patients With Dynamic Attributed Graphs in Online Health Communities." IEEE Transactions on Knowledge and Data Engineering 35, no. 2 (2022): 1831-1843."
