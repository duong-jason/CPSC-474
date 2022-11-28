# Distributed Random Forest
## Overview
**Random Forest** is a bagging (**b**oostrapping + **agg**rega**ing**) ensemble learning technique for both classification and regression tasks that utilizes multiple, independent, and different *(uncorrelatedness)* **decision trees** models. Each decision tree is trained on **bootstrap** samples *(random samples with replacement of the same size as the original dataset)* and **subspace** samples *(randomly selected subset of features from the original dataset)*. The predictions from all decision trees are then aggregated using some voting mechanism. Most common voting mechanisms include mode and mean for classification and regression tasks, respectively.

Random Forest addresses the issue of *overfitting* with a single decision tree. Having *random* decision tree models making predictions for the same task reduces variance because each model is trained on a subset of instances and features from the original dataset. In addition, random forests are likely to generalize better than a single decision tree as a consensus, more or less, yields a more accurate prediction than a single individual view.

## Goal
Although random forests have many advantages, one pitfall is that it is computationally expensive during the training phase. Our goal is to speedup training time through parallelization in the bootstrapping phase. Each decision tree in the forest is trained solely by one process. Process synchronization is required once all processes have been trained. Specifically, we use ```comm.Barrier()``` to pause the execution of a process until all processes have finished training. Aggregation is done using ```comm.allreduce(...)``` where all predictions are applied some voting mechanism to yield a majority prediction for all processes.

## Future Work
As of now, our decision tree implementation requires categorical feature values to run the *ID3 Algorithm*, implying non-existence of duplicate features on any path from the root. We plan to enable our decision tree models to make predictions for continuous feature values using *information gain* to find an optimal threshold for a feature when being tested. This implies that there may be duplicate features on any path from the root as it constitutes a binary search structure.

## How to Run
```bash
mpiexec -n <number_of_processes> python3 main.py
```
