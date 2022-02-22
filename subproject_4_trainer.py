import pandas as pd
import datetime
import json
import time
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

def main():
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

    data_path = "./dataset/ionosphere"

    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"), header=None).to_numpy()[:num_samples, :num_features]
    y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"), header=None).to_numpy()[:num_samples]

    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"), header=None).to_numpy()[:num_samples, :num_features]
    y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"), header=None).to_numpy()[:num_samples]

    print(X_train.shape)
    print(y_train.shape)

    if normalize_neg1_pos1:
        X_train = MyUtils.normalize_neg1_pos1(X_train)
        X_test = MyUtils.normalize_neg1_pos1(X_test)

    elif normalize_zero_one:
        X_train = MyUtils.normalize_0_1(X_train)
        X_test = MyUtils.normalize_0_1(X_test)

    # helper to format output file
    def save_training_output(training_output):
        base_path = "./output"
        join_str = "-"
        output_path = f"{base_path}/{datetime.datetime.now()}_GD_degree-{max_degree}_iterations-{training_iterations}_eta-{join_str.join([str(int) for int in eta_list])}_lam-{join_str.join([str(int) for int in lam_list])}_mbs-{join_str.join([str(mbs) for mbs in mini_batch_size_list])}.json"

        # post-process filename
        output_path = output_path.replace(":", "-").replace(" ", "_")
        with open(output_path, "w") as file:
            json.dump(training_output, file)

        print(f"\ntraining output saved at {output_path}")

    results = [] # results will hold dict of (degree, epochs, eta, lam, train_mse, test_mse, y_hat)

    for r in range(1, max_degree + 1):  # 1-based indexing
        print(f"degree {r}")

        print(f"\titerations {training_iterations}")

        for eta_val in eta_list:
            print(f"\t\teta {eta_val}")

            for lam_val in lam_list:
                print(f"\t\t\tlam {lam_val}")

                for mini_batch_size_val in mini_batch_size_list:
                    print(f"\t\t\t\tmini batch size {mini_batch_size_val}")

                    lr = LogisticRegression()

                    start = time.time()
                    train_mse, test_mse = lr.fit_metrics(X_train, y_train, X_test, y_test, lam=lam_val, eta=eta_val, iterations=training_iterations, degree=r, iteration_step=iteration_step, mini_batch_size=mini_batch_size_val)
                    end = time.time()

                    y_hat = lr.predict(X=X_test)

                    min_train_mse = min(train_mse)
                    min_test_mse = min(test_mse)

                    result = {
                        "degree": r,
                        "iterations": training_iterations,
                        "eta": eta_val,
                        "lam": lam_val,
                        "mini_batch_size": mini_batch_size_val,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                        "min_train_mse": min_train_mse,
                        "min_train_mse_iteration": train_mse.index(min_train_mse) * iteration_step,
                        "min_test_mse": min_test_mse,
                        "min_test_mse_iteration": test_mse.index(min_test_mse) * iteration_step,
                        "y_hat": list(y_hat.flatten()), # json doesnt like the nd-array
                        "train_time": (end - start) # training time in seconds
                    } 

                    results.append(result)

                    print(f"\t\t\t\tmini batch size {mini_batch_size_val} done;")

                print(f"\t\t\tlam {lam_val} done;")

            print(f"\t\teta {eta_val} done;")

        print(f"\titerations {training_iterations} done;")

        print(f"degree {r} done;")

    assert len(results) == max_degree * len(eta_list) * len(lam_list) * len(mini_batch_size_list)
    print(f"\nnumber of training runs: {len(results)}")

    # add metadata
    training_output = {
        "metadata": {
            "max_degree": max_degree,
            "iterations": training_iterations,
            "iteration_step": iteration_step,
            "eta_list": eta_list,
            "lam_list": lam_list,
            "mini_batch_size_list": mini_batch_size_list 
        },
        "results": results
    }

    # save training results
    save_training_output(training_output)
    notify_user(notify_when_done)

if __name__ == "__main__":
    main()
    