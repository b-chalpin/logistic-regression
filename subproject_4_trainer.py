import pandas as pd
import datetime
import json
import os

from code_misc.utils import MyUtils
from code_logistic_regression.logistic_regression import LogisticRegression

def notify_user(notify_when_done=False):
    """
        parameters
            notify_when_done: boolean, if True -> play sound
    """
    if notify_when_done:
        from playsound import playsound
        playsound('./misc/train_complete.mp3')
        
# training config
max_degree = 4
training_iterations = 100_000
iteration_step = 1000
minibatch_size = 100
eta_list = [0.01, 0.001, 0.0001]
lam_list = [0.01, 0.001, 0]
notify_when_done = False # option to notify user with audio when training is done

# dataset config
normalize_neg1_pos1 = False
normalize_zero_one = True
num_samples = None # set to None for all samples
num_features = None # set to None for all features

data_path = "./dataset/ionosphere"

X_train = pd.read_csv(os.path.join(data_path, "X_train.csv")).to_numpy()[:num_samples, :num_features]
y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).to_numpy()[:num_samples]
X_test = pd.read_csv(os.path.join(data_path, "X_test.csv")).to_numpy()[:num_samples, :num_features]
y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).to_numpy()[:num_samples]

print(X_train.shape)
print(y_train.shape)

if normalize_neg1_pos1:
    X_train = MyUtils.normalize_neg1_pos1(X_train)
    y_train = MyUtils.normalize_neg1_pos1(y_train)
    X_test = MyUtils.normalize_neg1_pos1(X_test)
    y_test = MyUtils.normalize_neg1_pos1(y_test)
    
elif normalize_zero_one:
    X_train = MyUtils.normalize_0_1(X_train)
    y_train = MyUtils.normalize_0_1(y_train)
    X_test = MyUtils.normalize_0_1(X_test)
    y_test = MyUtils.normalize_0_1(y_test)

# helper to format output file
def save_training_output(training_output):
    base_path = "./output"
    join_str = "-"
    output_path = f"{base_path}/{datetime.datetime.now()}_GD_degree-{max_degree}_iterations-{training_epochs}_eta-{join_str.join([str(int) for int in eta_list])}_lam-{join_str.join([str(int) for int in lam_list])}.json"

    # post-process filename
    output_path = output_path.replace(":", "-").replace(" ", "_")
    with open(output_path, "w") as file:
        json.dump(training_output, file)

    print(f"\ntraining output saved at {output_path}")

# perform training
LogisticRegression lr = LogisticRegression()
    
def fit(self, X_train, y_train, lam, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
    
save_training_output(gd_output)
notify_user(notify_when_done)
