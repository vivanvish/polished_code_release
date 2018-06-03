## Polished Code Release
Vishnu Nandakumar, DATA 558 SP.18


## Summary

This repository contains the code for my implementation of the Kernel based Support Vector Machine Classifier.
Support Vector Machines are powerfull tools, that can relies on finding the closest mathematical approximation of 
the distribution of your data. Hence it guarantees a mathematical base for the model.

The module currently supports two kernels: **rbf and polynomial.**. The module can be easily extended to include other
kernels as well. Currently the classifier is trained based on the One vs Rest strategy. Extending this to include the 
One vs One strategy is part of future work.


The model uses **Fast Gradient Descent**, along with **Backtracking Line Search** algorithm to find the optimum learning rate.

## Instructions
Files:
  KernelSVM.py : Implementation of the Kernel SVM classifier. 
  Example.py: Sample code on how to use the classifier.
  
Feel free to use or distribute the code. The package can be used by:
1. Clone the repository
  ```
  git clone  https://github.com/vivanvish/polished_code_release.git
  ```
2. Install dependencies.
  ```
  pip install -r requirements.txt
  ```
