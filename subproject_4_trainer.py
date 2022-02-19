import pandas as pd
import datetime
import json
import time
import os

from code_misc.utils import MyUtils
from code_logistic_regression.logistic_regression import LogisticRegression

def main:
    def notify_user(notify_when_done=False):
        """
            parameters
                notify_when_done: boolean, if True -> play sound
        """
        if notify_when_done:
            from playsound import playsound
            playsound('./misc/train_complete.mp3')

    # training config
    max_degree = 1
    training_iterations = 100_000
    iteration_step = 1000
    mini_batch_size = 100
    eta_list = [0.01]
    lam_list = [0.01]
    notify_when_done = False # option to notify user with audio when training is done

    # dataset config
    normalize_neg1_pos1 = False
    normalize_zero_one = True
    num_samples = None # set to None for all samples
    num_features = None # set to None for all features

    data_path = "./dataset/ionosphere"

    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"), header=None).to_numpy()[:num_samples, :num_features]
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"), header=None).to_numpy()[:num_samples]

    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"), header=None).to_numpy()[:num_samples, :num_features]
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"), header=None).to_numpy()[:num_samples]

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
        output_path = f"{base_path}/{datetime.datetime.now()}_GD_degree-{max_degree}_iterations-{training_iterations}_eta-{join_str.join([str(int) for int in eta_list])}_lam-{join_str.join([str(int) for int in lam_list])}.json"

        # post-process filename
        output_path = output_path.replace(":", "-").replace(" ", "_")
        with open(output_path, "w") as file:
            json.dump(training_output, file)

        print(f"\ntraining output saved at {output_path}")

    # perform training
    lr = LogisticRegression()

    results = [] # results will hold dict of (degree, epochs, eta, lam, train_mse, test_mse, y_hat)

    for r in range(1, max_degree + 1):  # 1-based indexing
        print(f"degree {r}")

        print(f"\titerations {training_iterations}")

        for eta_val in eta_list:
            print(f"\t\teta {eta_val}")

            for lam_val in lam_list:
                print(f"\t\t\tlam {lam_val}")

                start = time.time()
                train_mse, test_mse = ([0], [0])

                lr.fit(X=X_train, y=y_train, lam=lam_val, eta=eta_val, iterations=training_iterations, mini_batch_size=mini_batch_size)

                # implement helper function
                # train_mse, test_mse = lr.fit_metrics(X=X_train, y=y_train, X_test=X_test, y_test=y_test, epochs=training_epochs, eta=eta_val, degree=r, lam=lam_val, epoch_step=epoch_step)
                end = time.time()

                y_hat = lr.predict(X=X_test)

                min_train_mse = min(train_mse)
                min_test_mse = min(test_mse)

                result = {
                    "degree": r,
                    "iterations": training_iterations,
                    "eta": eta_val,
                    "lam": lam_val,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "min_train_mse": min_train_mse,
                    "min_train_mse_epoch": train_mse.index(min_train_mse) * iteration_step,
                    "min_test_mse": min_test_mse,
                    "min_test_mse_epoch": test_mse.index(min_test_mse) * iteration_step,
                    "y_hat": list(y_hat.flatten()), # json doesnt like the nd-array
                    "train_time": (end - start) # trainng time in seconds
                } 

                results.append(result)

                print(f"\t\t\tlam {lam_val} done;")

            print(f"\t\teta {eta_val} done;")

        print(f"\titerations {training_iterations} done;")

        print(f"degree {r} done;")

    assert len(results) == max_degree * len(eta_list) * len(lam_list)
    print(f"\nnumber of training runs: {len(results)}")

    # add metadata
    training_output = {
        "metadata": {
            "max_degree": max_degree,
            "training_epochs": training_iterations,
            "iteration_step": iteration_step,
            "eta_list": eta_list,
            "lam_list": lam_list
        },
        "results": results
    }

    # save training results
    save_training_output(training_output)
    notify_user(notify_when_done)

if __name__ == "__main__":
    main()
    