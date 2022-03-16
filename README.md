
# Backdoor Learning Curves: Explaining BackdoorPoisoning Beyond Influence Functions

Here we provide the code used to run the experiments analyzed in "Backdoor Learning Curves: Explaining BackdoorPoisoning Beyond Influence Functions" (submitted to Neurocomputing Journal).

Here is a conceptual figure of our work. In summary, when decreasing the regularization parameter $\lambda$, the classifier learns the backdoor faster.

<!-- ![Backdoor learning](./animation.gif) -->

<p align="center">
  <img width="460" height="420" src="./animation.gif">
</p>

Project structure:

- `src/`
  - `attacks`, source code for backdoor poisoning and influence function;
  - `classifiers`, torch modules for features extraction with pre-trained networks, and data preprocessing;
  - `experiments`, script used to run our experiments with binary MNIST and CIFAR10;
  - `test`, contains script to test the backdoor effectiveness and explain the backdoor effects;
  - `utilities`', contains utility functions for data loading, attack evaluation and plot settings;
- `results/`
  - `mnist`, contains figures and results for MNIST;
  - `cifar`, contains figures and results for CIFAR10;
  - `imagenette`, contains figures and results for Imagenette;
  
## Installation

Import the conda environment:

```bash
conda env create -f env.yml 
```

Then, we need to activate the conda env with:

```bash
conda activate backdoor_curves
```

## Run experiments

### Backdoor Learning Slope

Our experiments for analyzing the backdoor learning slope involve three pairs of dataset from MNIST, CIFAR10 and Imagenette. For each of them, we test:
    - Support Vector Machine;
    - Logistic Classifier;
    - Ridge Classifier;
    - Support Vector Machine with RBF kernel;
    - ResNet18

Different hyperparameters have been chosen to test their robustness against the backdoor poisoning attack for each of them.
The user may replicate our experiments by running the commands in `run_cifar.sh`, `run_imagenette.sh` and `run_mnist.sh`.

For example, by running from the console the command:

```bash
./run_mnist.sh all
```

The script will run all the experiments for the three pairs of dataset from MNIST. If we aim to restrict our analysis to only one pair, we may run:

```bash
./run_mnist.sh mnist_30
```

This will run the experiments only for MNIST 3 vs 0. The same thing is valid for CIFAR10 and Imagenette.

```bash
./run_cifar.sh cifar_60
```

```bash
./run_imagenette.sh imagenette_60
```

Where for CIFAR10 6-0, 2-5 and 0-9 refers to airplance vs. frog, bird vs. dog and airplane vs. truck respectively. While for Imagenette 6-0, 2-5 and 0-9 refers to tench vs. truck, player vs. church and tench vs. parachute.

For Neural Networks we used the script in `src/experiments/nn/backdoor_slope.py`. To test the backdoor learning slope for Resnet18 and Resnet50 run: 

```bash
python src/experiments/nn/backdoor_slope.py
```

## Incremental Backdoor Learning

In order to replicate our studies on the backdoor learning curves and our analysis on the impact
of backdoor poisoning on learning parameter, it is necessary to run `incremental.py` 
with the following command:

```bash
python incremental_curves.py 
```

By default we used `torch.nn.DataParallel` to allow multi-gpus computations. However, this can be removed and nets can be moved to any device.

## Explaining Backdoor Predictions

With these experiments, we aim to interpret the decision function of the poisoned classifiers. To replicate our analysis with MNIST and CIFAR10, use the following commands:

```bash
python src/test/explain_mnist.py 
```

```bash
python src/test/explain_cifar10.py
```

```bash
python src/test/explain_imagenette.py
```
