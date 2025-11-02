# (NeurIPS2025 $\textcolor{blue}{SPOTLIGHT}$) Higher-Order-Differential-Topology-aware-GNN 
The code of the proposed model DEC-HOGNN on the benchmark 2D-electrostatics is offered, as shown in Table 2 in our paper.  The code doesn't need any extra dependency except for popular libraries like `pytorch`,`numpy` and some python standard libraries. 

### Instructions
Class `HOGNNDataset` in `datagen2D.py` has encapsulated all stuff that DEC-HOGNN requires (like adjacencies, volumes, boundary markers, whitney elements etc). The training, validation and test datasets have been already partitioned in ./data. The training code lies in `main.py` and some auxilary classes are in `basics.py`. 


### Quick Start
A trained model within 500 epochs is provided. You can load the parameters and run the test function by simply typing in the command:

`python main.py --gpu=0 --mode=1`

You can also choose to train the model by: 

` python main.py --gpu=0 --mode=0`

You can cite our paper by (.bibtex): 
```
@inproceedings{
liao2025boundaryvalue,
title={Boundary-Value {PDE}s Meet Higher-Order Differential Topology-aware {GNN}s},
author={Yunfeng Liao and Yangxin Wu and Xiucheng Li},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=PluDA8DEar}
}
```