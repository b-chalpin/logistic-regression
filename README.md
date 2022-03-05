# Logistic Regression By Hand

## Description

Using `numpy` and native Python, implement a Logistic Regression classification model. The model is able to be trained using a Closed-form method, or Gradient Descent method.

## Environment Installation

##### Using Conda

To create the virtual environment and install required Python packages, run the following commands in the terminal:

```
$conda env create -f environment.yml
$conda activate cscd496-prog5-bchalpin
```

##### Without Conda

If you do not have Conda installed, the packages may still be installed using the following command:

```
$pip install -r requirements.txt
```

## Linear Regression Training

For training, first open `subproject_4_trainer.py` in a code editor of your choice. Adjust the hyperparameter configuration however you like. These will look like

```
# training config
max_degree = 2
training_iterations = 10_000
iteration_step = 500
eta_list = [0.1, 0.01, 0.001]
lam_list = [0.01, 0.001, 0]
mini_batch_size_list = [50, 100, 200]
notify_when_done = False # option to notify user with audio when training is done

# dataset config
normalize_neg1_pos1 = True
normalize_zero_one = False
num_samples = None # set to None for all samples
num_features = None # set to None for all features
```

Lastly, train the model by exeuting the following command.

```
$python subproject_4_trainer.py
```

## Visualization

Copy the file names of the training output from the previous section. Within `subproject_4.ipynb` replace the following code with your new output file names:

```
...
# change the filenames below accordingly.
### note that when running both GF and CF visualization, hyperparameter configs MUST match
sgd_output_filename = f"{sgd_output_base_path}/2022-02-21_20-52-38.446823_GD_degree-2_iterations-10000_eta-0.1-0.01-0.001_lam-0.01-0.001-0_mbs-50-100-200.json"
sgd_training_output = open_training_output_file(sgd_output_filename)
...
```

### Author

Blake Chalpin [b-chalpin](https://github.com/b-chalpin)
