# Interferometer alignment
This is the code of paper: Aligning an optical interferometer with beam divergence control and continuous action space.

Paper: https://arxiv.org/pdf/2107.04457.pdf

Examples of alignment:

<!-- ![fig1](https://github.com/Stepan-Makarenko/RL_interferometer_alignment/blob/main/) -->
<img src="/media/fig1.gif" width="49%" height="49%"/> <img src="/media/fig2.gif" width="49%" height="49%"/>
<!-- ![fig2](https://github.com/Stepan-Makarenko/RL_interferometer_alignment/blob/main/media/fig2.gif) -->

## Requirements
The working simulation of the Mach-Zehnder interferometer could be installed from https://github.com/dmitrySorokin/interferobotProject

python=3.7.9
torch=1.6.0
gym=0.12.1
numpy=1.19.1
wandb=0.10.15

## Training
The whole training process is described in TD3_training.ipynb notebook.
