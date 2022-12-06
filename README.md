# Scene Transformer

Unofficial proof-of-concept implementation of  
**Scene Transformer: A unified architecture for predicting multiple agent trajectories** (https://arxiv.org/abs/2106.08417)  
paper by Google Brain. 

We implement their model and apply to [SDC](https://research.yandex.com/publications/shifts-a-dataset-of-real-distributional-shift-across-multiple-large-scale-tasks) dataset. Only factorized self-attention part of the model is implemented, which is responsible for trajectory prediction based on time and agents in a scene. Cross-attetion part accountable for road graph is missing. 

The structure of repo is the following:
 - ```scenetransformer.py```: model impltemetation,
 - ```sdcdataset.py```: dataloader implementation for SDC dataset
 - ```sdc_train_loop.py```: train loop + loss implementation
