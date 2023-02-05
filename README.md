# Differential Privacy in Multiagent Systems

This python project contains code associated with the paper *Differential Privacy in Cooperative Multiagent Planning*.

## Requirements
This project requires Python 3 with the following packages:
- Numpy
- matplotlib
- cvxpy
- Mosek

### Installation instructions using Anaconda
- Download and install Anaconda from (https://www.anaconda.com/products/individual-d).
- To create a new virtual environment run: 
  - >conda create -n mac python=3.9
- Activate the virtual environment:
  - >conda activate dpmas
- Install the necessary packages:
  - >conda install numpy
  - >conda install matplotlib
- Follow the OS-specific installation instructions for cvxpy (https://www.cvxpy.org/install/) and Mosek (https://docs.mosek.com/9.2/pythonapi/install-interface.html).

## Running Experiments

To generate a minimum dependency policy, set the current directory to experiments/, point the "config_file" variable in "experiments/synthesize_minimum_dependency_policy.py" to the desired YAML experiment configuration file, and run the following command:
```
python synthesize_minimum_dependency_policy.py
```

This will run the experiment, and save the results to a pickle file stored in experiments/results. The experiment setup and hyperparameters can be changed by editing the YAML config files in experiments/configuirations.

## Visualizing the Experiment Results
Code to visualize and analyze the results of the experiments can be found in the plotting/ directory.