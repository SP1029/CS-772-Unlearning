# CS772 Project: Zero-Shot Machine Unlearning

## Overview
This repository contains the work done as part of the CS772 course project during the 6th semester.

## Project Team
- Krish @flintmarko
- Siddharth @sid-kal
- Shubham @SP1029
- Labajyoti @labajyoti21
- Ashutosh @ashutoshkr4458

## Approach
The project builds upon the work presented in the seed paper "Zero-shot Machine Unlearning" by V.S. Chundawat et al.

1. **Error Minimization-Maximization Noise**

2. **Gated Knowledge Transfer**

## Repository Structure
  - `deep-inversion.ipynb`: Generating images of MNIST numbers from the trained model using Deep Inversion technique.
  - `entropy.ipynb`: Implementing the entropy criteria for filtering generated data points.
  - `no-attn-student.ipynb`: Removing Attention from student loss.
  - `deep-retrain.ipynb`: Retraining a new model using the images generated by Deep Inversion.

## Code tweaked from
- https://github.com/ayushkumartarun/zero-shot-unlearning
- https://github.com/NVlabs/DeepInversion/tree/master
