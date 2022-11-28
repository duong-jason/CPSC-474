# Distributed Random Forest
## Overview
**Random Forest** is a bagging (**b**oostrapping + **agg**rega**ing**) ensemble learning technique for both classification and regression tasks that utilizes multiple, independent, and different *(uncorrelatedness)* **decision trees** models. Each decision tree is trained on a **bootstrap** sample *(random samples with replacement of the same size as the original dataset)* and **subspace** samples *(randomly selected subset of features from the original dataset)*. The predictions from all decision trees are then aggregated using some voting mechanism. Most common voting mechanisms include mode and mean for classification and regression tasks, respectively.

Random Forest addresses the issue of *overfitting* with a single decision tree. Having *random* decision tree models making predictions for the same task reduces variance because each model is trained on a subset of instances and features from the original dataset. In addition, random forests are likely to generalize better than a single decision tree as a consensus, more or less, tend to yield more accurate predictions than a single individual view.

## Goal
Although random forests have many advantages, one pitfall is that it is computationally expensive during the training phase. Our goal is to speedup training time through parallelization through distributed computing using the MPI framework. Each decision tree in the forest is trained solely by one process. Process synchronization is required once all processes have fully constructed their decision tree. Specifically, we use ```comm.Barrier()``` to pause the execution of a process until all processes have finished training. Aggregation is done using ```comm.allreduce(...)``` where all predictions are applied some voting mechanism to yield a majority prediction for all processes.

## Methodology
We use the *ID3 (Iterative Dichotomiser 3) algorithm* to induce a decision tree. The ID3 algorithm recursively grows the tree in a depth-first search manner by finding the optimal feature for which to split and repeating until some base case is reached and a leaf node is created. The splitting criteria is based on **information gain**, which is the reduction of uncertainty before and after the split. We use two standard approaches to measure impurity: *entropy* and *gini*, both which are implemented. Essentially, the feature that provides the highest information gain among the possible feature levels is chosen for a particular path and a interior node is created. Each recursive call partitions the current dataset according to the feature values for the respective level.

For example: The $Occupation$ feature, with $Lawyer$, $Teacher$, and $Doctor$ as the possible levels, has the highest information gain and is chosen from dataset $\mathcal{D}$ for which to split. Each branch is recursively grown from level $d\in\\lbrace Lawyer, Teacher, Doctor\rbrace$ with partitioned dataset $\mathcal{D}[Occupation=d]$ containing all instances for which $Occupation=d$.

In correspondence to overfitting, we enable pre-pruning criterions for decision trees where the algorithm stops growing on a path after some stage. These criterions may be when the depth of the tree, the number of instances in a partitioned dataset or the amount of information gain exceeds a pre-defined threshold.

## Future Work
As of now, our decision tree implementation requires categorical feature values to run the *ID3 Algorithm*, implying non-existence of duplicate features on any path from the root. We plan to enable our decision tree models to make predictions for continuous feature values using *information gain* to find an optimal threshold for a feature when being tested. This implies that there may be duplicate features on any path from the root as it resembles a binary search structure.

## How to Run
```bash
mpiexec -n <number_of_processes> python3 main.py
```
