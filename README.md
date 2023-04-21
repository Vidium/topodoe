# Topodoe

Design of Experiment package for Python, based on topological analysis.
This method is applicable on an ensembl of executable Gene Regulatory Network (GRN) models. This package provides tools for identifying the next best experiment to perform to reduce the set of candidate GRNs as much as possible.

Steps:
1. Topological analysis - in this step we reduce the number of candidate perturbations to test

2. Simulation of perturbation (to run using you GRN model, not done by TopoDoE) and ranking - in this step we identify the most informative perturbation from in-silico simulations
    
3. Acquisition of biological data - in this step we perform the selected perturbation in-vitro to obtain biological data to compare to simulations
    
4. Selection of a subset of GRNs best fitting the data - in this step - in this step we select a subset of the candidate GRNs which simulation correctly predicted the outcome of the pertubation

We provide utility classes to store GRN models and create plots (Grn and GrnCollection).
