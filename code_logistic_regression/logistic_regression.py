########## Blake Chalpin 00864973

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent

import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils

def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
        v_sigmoid = np.vectorize(_sigmoid)
        return v_sigmoid(s)
    
def _sigmoid(s):
    ''' s: a real number
        return: the sigmoid function value of the input signal s
    '''
    return 1 / (1 + np.exp(-s))

class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1
        
    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''
        self.degree = degree
        X = MyUtils.z_transform(X, degree=self.degree)
        
        # if we are not doing SGD, minibatch size will be size N (# of all samples)
        if not SGD:
            mini_batch_size = X.shape[0]
        
        self._fit_sgd(X=X, y=y, lam=lam, eta=eta, iterations=iterations, mini_batch_size=mini_batch_size)
    
    def _fit_sgd(self, X, y, lam = 0, eta = 0.01, iterations = 1000, mini_batch_size = 1):
        ''' Perform Stochastic Gradient Descent training. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                mini_batch_size: the size of each mini batch size.  
        '''
        X_bias = self._add_bias_column(X)
        _, d = X_bias.shape
        self._init_w_vector(d)
        
        mini_batch_list = self._generate_mini_batches(X_bias, y, mini_batch_size)
        NUM_MINI_BATCHES = len(mini_batch_list)

        mini_batch_index = 0 # index to keep track of minibatch we are on
        while iterations > 0:
            X_mini, y_mini = mini_batch_list[mini_batch_index]
            n_mini, _ = X_mini.shape

            # print(f"Batch number index: {mini_batch_index}")
            # print(f"\tMini batch {mini_batch_index}: X: {mini_batch_list[mini_batch_index][0].shape}, y: {mini_batch_list[mini_batch_index][0].shape}")

            s = y_mini * (X_mini @ self.w)
            self.w = (eta / n_mini) * ((y_mini * _v_sigmoid(-s)).T @ X_mini).T + (1 - (2 * lam * eta / n_mini)) * self.w

            iterations -= 1
            mini_batch_index = (mini_batch_index + 1) % NUM_MINI_BATCHES # wrap around to index 0 when at the end
    
    def _generate_mini_batches(self, X, y, mini_batch_size):
        ''' Performs mini batching. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                mini_batch_size: the size of each mini batch size.  
            Returns:
                [(X', y')_1, ... (X', y')_N]: List of tuples for (X, y), each representing a minibatch
        '''
        n, _ = X.shape
        NUM_MINI_BATCHES = math.ceil(n / mini_batch_size)

        mini_batch_list = []

        for i in range(NUM_MINI_BATCHES):
            mini_batch_start_index = i * mini_batch_size
            mini_batch_end_index = (i + 1) * mini_batch_size

            mini_batch = (X[mini_batch_start_index : mini_batch_end_index],
                          y[mini_batch_start_index : mini_batch_end_index])

            mini_batch_list.append(mini_batch)
            
        assert len(mini_batch_list) == NUM_MINI_BATCHES
        # assert that the sum of all minibatch lengths equals original set size
        assert sum(len(mini_batch[0]) for mini_batch in mini_batch_list) == n
        return mini_batch_list

    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        Z = MyUtils.z_transform(X, degree=self.degree)  # Z-transform to match self.w dimension
        Z_bias = self._add_bias_column(Z)

        return _v_sigmoid(Z_bias @ self.w)  # it is assumed that self.w is already trained
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        y_hat = self.predict(X)
        
        misclassified = 0
        for (y_pred, y_label) in zip(y_hat, y):
            if (y_pred > 0.5 and int(y_label) == -1) or (y_pred <= 0.5 and int(y_label) == 1):
                misclassified += 1

        return misclassified
    
    def _error_z(self, Z, y):
        """ An internal helper method to calculate MSE for already
            Z-tranformed samples
            parameters:
                Z: n x d matrix of Z-transformed samples (INCLUDING BIAS)
                y: n x 1 matrix of labels
            return:
                the MSE for this test set (Z,y) using the trained model
        """
        y_hat = Z @ self.w

        return self._calculate_mse(y_hat, y)

    def _init_w_vector(self, d):
        """ If self.w does not exist, it is initialized
            parameters:
                d: scalar, representing number of features in our Z-tranformed samples
        """
        self.w = np.zeros((d, 1))

    def _add_bias_column(self, X):
        """ parameters:
                X: n x d matrix of future samples
            return:
                X: n x (d+1) matrix, with added bias column
        """
        return np.insert(X, 0, 1, axis=1)
