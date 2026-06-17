# Weights & Biases Hands-on

A tutorial repo for learning the basics of [Weights & Biases](https://wandb.ai/) using a simple `scikit-learn` SVM classifier. This tutorial covers the following:

- Creating a W&B account and API key. 
- Login to W&B account.
- Train and log a scikit-learn model.
- Run a W&B sweep over model hyperparameters.

---

## 1. Create a W&B account

Create a free account at:

<https://wandb.ai/site>

Then follow the W&B quickstart guide to generate your API key:

<https://docs.wandb.ai/models/quickstart#personal-api-key>

Keep your API key somewhere safe. You will need it when logging in from the terminal.

---

## 2. Set up python environment & Log in via API key

You can use either `uv` or `conda`.

---

### Option A: using `uv`

Create the environment and install dependencies:
```bash
uv sync
```

Log in to W&B:
```bash
uv run wandb login
```

Paste your W&B API key when prompted.

---

### Option B: using `conda`

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate wandb-svm-tutorial
```

Log in to W&B:

```bash
wandb login
```

Paste your W&B API key when prompted.

---

## 3. Quickstart: check W&B logging works

Start with the quickstart script from the W&B website to check things are running correctly:

```bash
uv run python quickstart.py
```

Or, with conda:

```bash
python quickstart.py
```

This should create a simple W&B run. Open the link printed in the terminal to view your logged metrics.

---

## 4. Run one SVM training job

Next, run a single SVM classifier using the default config:

```bash
uv run train.py
```

Or, with conda:

```bash
python train.py
```

This trains one model, logs metrics to W&B, and saves a decision-boundary plot.

The default config controls the dataset and model settings:

```yaml
dataset:
  n_samples: 1000
  noise: 0.30
  test_size: 0.30
  random_state: 1

model:
  kernel: rbf
  C: 3.0
  gamma: 1.0
  degree: 3
```

---

## 5. What is a W&B sweep?

A sweep runs many training jobs with different hyperparameters. The sweep is defined in the ```config/sweep.yaml``` file, where we choose:
- python file used for training
- type of hyperparameter search
- space of hyperparamters to search
- metric to optimise


We defined the program (name of the python file to run) and the name of the project.
```yaml
program: train.py
project: wandb-tutorial
```

The method (type of hyperparameter search) is set to ```grid```. 
```yaml
method: grid
```
Other choices are ```random``` for selecting random combinations, or ```bayesian``` which uses Bayesian optimisation for selecting hyperparameters.

Here we choose the metric we want, and which way to optimise (here we are maximising). The name needs to match a metric that is logged to W&B by the program (here, train.py).

```yaml
metric:
  name: test_accuracy
  goal: maximize
```

Finally the space of parameters to search - here we just look at a simple 1D grid search over regularisation parameter. 
```yaml
parameters:
  gamma:
    values: [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
```

## 6. Running a W&B sweep

First, create the sweep:

```bash
wandb sweep configs/sweep.yaml
```

W&B will print a command that looks something like this:

```bash
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
```

<b>This will not run the sweep yet</b>. To run the sweep we need to use the command:

```bash
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
```

View the sweep in your W&B dashboard.