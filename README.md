# Heterogeneous Datasets for Federated Survival Analysis Simulation

This repo contains three algorithms for constructing realistic federated datasets for survival analysis.
Each algorithm starts from an existing non-federated dataset and assigns each sample to a specific client in the federation.
The algorithms are
* ```uniform_split```: assigns each sample to a random client with uniform probability;
* ```quantity_skewed_split```: assigns each sample to a random client according to the Dirichlet distribution [3, 4];
* ```label_skewed_split```: assigns each sample to a time bin, then assigns a set of samples from each bin to the clients according to the Dirichlet distribution [3, 4].

For more information, please take a look at our [paper](https://arxiv.org/abs/2301.12166) [1].

## ⚙️ Installation

Federated Survival Datasets is built on top of [numpy](https://numpy.org/) and 
[scikit-learn](https://scikit-learn.org/stable/). 
To install those libraries you can run

```pip install -r requirements.txt```

To import survival datasets in your project, we strongly recommend [SurvSet](https://github.com/ErikinBC/SurvSet) [2], a comprehensive collection of more than 70 survival datasets.

## 🛠️ Usage

```python
import numpy as np
import pandas as pd

from federated_survival_datasets import label_skewed_split

# import a survival dataset and extract the input array X and the output array y
df = pd.read_csv("metabric.csv")
X = df[[f"x{i}" for i in range(9)]].to_numpy()
y = np.array([(e, t) for e, t in zip(df["event"], df["time"])], dtype=[("event", bool), ("time", float)])

# run the splitting algorithm
client_data = label_skewed_split(num_clients=8, X=X, y=y)

# check the number of samples assigned to each client
for i, (X_c, y_c) in enumerate(client_data):
    print(f"Client {i} - X: {X_c.shape}, y: {y_c.shape}")
```

We provide an [example notebook](https://github.com/archettialberto/federated_survival_datasets/blob/main/example/example_usage.ipynb) to illustrate the proposed algorithms.
It requires [scikit-survival](https://scikit-survival.readthedocs.io/en/stable/index.html#), [seaborn](https://seaborn.pydata.org/), and [pandas](https://pandas.pydata.org/).

## 📕 Bibtex Citation
```
@article{archetti2023heterogeneous,
  title={Heterogeneous Datasets for Federated Survival Analysis Simulation},
  author={Archetti, Alberto and Lomurno, Eugenio and Lattari, Francesco and Martin, Andr{\'e} and Matteucci, Matteo},
  journal={arXiv preprint arXiv:2301.12166},
  year={2023}
}
```

## 📚 References

[1] Archetti, A., Lomurno, E., Lattari, F., Martin, A., & Matteucci, M. (2023). Heterogeneous Datasets for Federated Survival Analysis Simulation. arXiv preprint arXiv:2301.12166.

[2] Drysdale, E. (2022). SurvSet: An open-source time-to-event dataset repository. arXiv preprint arXiv:2203.03094.

[3] Hsu, T. M. H., Qi, H., & Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335.

[4] Li, Q., Diao, Y., Chen, Q., & He, B. (2022, May). Federated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) (pp. 965-978). IEEE.
