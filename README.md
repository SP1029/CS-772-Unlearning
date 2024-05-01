# CS772 Project: Zero-Shot Machine Unlearning

## Overview
This repository contains the work done as part of the CS772 course project under the guidance of Prof. Piyush Rai during the 6th semester. The project focuses on the concept of Zero-Shot Machine Unlearning, which aims to remove specific information from a pre-trained machine learning model without the need for retraining from scratch.

## Motivation
With the increasing emphasis on data privacy and regulations like GDPR, there is a growing demand for mechanisms that allow users to request the deletion of their data from machine learning models. However, retraining models from scratch is often time-consuming and resource-intensive. Zero-Shot Machine Unlearning provides a faster and more efficient approach to remove specific data from an already trained model.

## Project Team
This project was a collaborative effort by the following team members:

- Krish @flintmarko
- Siddharth @sid-kal
- Shubham @SP1029
- Labajyoti @labajyoti21
- Ashutosh @ashutoshkr4458

## Approach
The project builds upon the work presented in the seed paper "Zero-shot Machine Unlearning" by V.S. Chundawat et al. We investigated and implemented two main approaches:

1. **Error Minimization-Maximization Noise**: This approach involves training noise vectors for the classes to be forgotten (forget classes) and retained (retain classes). These noise vectors are then used to update the model parameters, effectively unlearning the forget classes.

2. **Gated Knowledge Transfer**: This method utilizes knowledge distillation to transfer the knowledge from the original model (teacher) to a new model (student), while ensuring that the student does not learn the forgotten classes.

## Contributions
In addition to implementing the approaches from the seed paper, our team made the following contributions:

- Explored the use of entropy filtering to improve the quality of generated data points for training the student model.
- Identified and addressed an inconsistency between retraining and unlearning scenarios, leading to improved performance.
- Incorporated DeepInversion techniques to generate high-quality data samples, enhancing the training process for the student model.

## Repository Structure
The repository is organized as follows:

- `docs/`: Contains the project report and related documentation.
- `src/`: Contains the source code for implementing the Zero-Shot Machine Unlearning approaches.
- `data/`: Placeholder for any data files used in the project.
- `experiments/`: Contains scripts and results for various experiments conducted during the project.

## Getting Started
To get started with this project, please refer to the `docs/` directory for detailed instructions on setting up the environment, running the code, and reproducing the experiments.

## Acknowledgments
We would like to express our gratitude to Prof. Piyush Rai for his guidance and support throughout the project. We also acknowledge the authors of the seed paper for their valuable contributions to the field of Machine Unlearning.
