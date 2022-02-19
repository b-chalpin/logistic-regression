##### >>>>>> Blake Chalpin 00864973

# Various tools for data manipulation. 

import numpy as np
import math

class MyUtils:
    def z_transform(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
        if degree == 1:
            return X

        _, d = X.shape # get the X-space dimension

        Z = X.copy() # retain the original dataset X

        r = degree

        B = [] # B holds the size of each bucket; B[i] = size of bucket i
        for i in range(r):
            B.append(math.comb(i + d, d - 1)) # 0-based indexing so not i + d -1

        d_prime = np.sum(B) # total number of columns in Z-space
        l = np.arange(d_prime, dtype=int) # initalize l for every bucket element

        # locate the start col index and end col index of previous bucket. this will be q and p
        q = 0 # total size of all buckets BEFORE the previous bucket B_{i-2}
        p = d # the size of the all previous buckets

        for i in range(1, r): # for each next bucket
            g = p # index of the new feature to add in bucket_i

            for j in range(q, p): # go through the previous bucket
                head = l[j] # get the head element for our current index to be added

                for k in range(head, d): # all indices of columns in X matrix; we will take them one-by-one 
                    new_feature = (Z[:, j] * X[:, k]).reshape(-1, 1) # produce a new element into the current bucket
                    Z = np.append(Z, new_feature, axis=1)
                    l[g] = k # update the head vector for our new feature column
                    g += 1 # increment our new feature index

            q = p # update q to be the end of the previous bucket
            p += B[i] # increment p by the previous bucket size

        assert(Z.shape[1] == d_prime) # assert our final Z-space matrix has correct dimensionality

        return Z
    
    ## below are the code that your instructor wrote for feature normalization. You can feel free to use them
    ## but you don't have to, if you want to use your own code or other library functions. 

    def normalize_0_1(X):
        ''' Normalize the value of every feature into the [0,1] range, using formula: x = (x-x_min)/(x_max - x_min)
            1) First shift all feature values to be non-negative by subtracting the min of each column 
               if that min is negative.
            2) Then divide each feature value by the max of the column if that max is not zero. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [0,1]
        '''

        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            gap = col_max - col_min
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_min) / gap
            else:
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]
        
        return X_norm

    def normalize_neg1_pos1(X):
        ''' Normalize the value of every feature into the [-1,+1] range. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [-1,1]
        '''

        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            col_mid = (col_max + col_min) / 2
            gap = (col_max - col_min) / 2
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_mid) / gap
            else: 
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]

        return X_norm
