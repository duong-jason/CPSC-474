# Distributed Random Forest
## Overview
**Random Forest** is an bagging ensemble learning technique for both classificaiton and regression tasks that utilizes multiple, independent, and different *(uncorrelatedness)* **decision trees** models. Firstly, each decision tree is trained on **bootstrap** samples *(random samples with replacement of the same size of the original dataset)* and **subspace** samples *(randomly selected subset of features from the original dataset)*. Secondly, the predictions from all decision trees are aggregated using some voting mechanism. Most common voting mechanisms include mode and mean for classification and regression tasks respectively.

Random Forest addresses the issue of *overfitting* with a single decision tree. Thus, having *random* decision tree models making predictions for the same task reduces variance because each model is trained on a subset of instances and features. In addition, random forests is likely to generalize better than a single decision tree as a consensus, more or less, yields a more accurate prediction than a single individual view.

## Goal
Although random forests has many advantages, it is computationally expensive during the training phase. Our goal is to speedup training time through parallelization in the boostrapping phase. Each decision tree in the forest is trained solely by one process. Process synchronization is required once all processes has been trained. Specifically, we use ```comm.Barrier()``` to pause the execution of the algorithm until all processes have finished training. Aggregation is done using ```comm.allreduce(...)``` where all predictions are gathered some voting mechanism is applied, yielding a majority prediction.

## How to Run
```bash
mpiexec -n <number_of_processes> python3 main.py
```
