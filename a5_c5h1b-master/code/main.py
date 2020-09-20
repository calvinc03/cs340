import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import fnc

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2
from pca import PCA, AlternativePCA, RobustPCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")
        
        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=1)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        kernelpoly = kernelLogRegL2(kernel_fun=logReg.kernel_poly, lammy=0.01)
        kernelpoly.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(kernelpoly.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(kernelpoly.predict(Xtest) != ytest))

        utils.plotClassifier(kernelpoly, Xtrain, ytrain)
        utils.savefig("logRegPolyKernel.png")

        kernelRBF = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=0.01)
        kernelRBF.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(kernelRBF.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(kernelRBF.predict(Xtest) != ytest))

        utils.plotClassifier(kernelRBF, Xtrain, ytrain)
        utils.savefig("logRegRBFKernel.png")

    elif question == "1.2":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=0)

        TError = np.inf
        VError = np.inf
        RBFT = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=10)
        RBFV = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=10)
        lamb1 = 0
        lamb2 = 0
        sig1 = 0
        sig2 = 0
        for i in range(-2,3):
            for j in range(-4,1):
                kernelRBF = kernelLogRegL2(kernel_fun=logReg.kernel_RBF, lammy=10**j, sigma = 10**i)
                kernelRBF.fit(Xtrain, ytrain)
                CTerror = np.mean(kernelRBF.predict(Xtrain) != ytrain)
                CVerror = np.mean(kernelRBF.predict(Xtest) != ytest)
                if CVerror < VError:
                    RBFV = kernelRBF
                    lamb1 = 10**j
                    sig1 = 10**i
                if CTerror < TError:
                    RBFT = kernelRBF
                    lamb2 = 10**j
                    sig2 = 10**i

        print(lamb1)
        print(sig1)
        print(lamb2)
        print(sig2)

        utils.plotClassifier(RBFT, Xtrain, ytrain)
        utils.savefig("RBFKernelT.png")

        utils.plotClassifier(RBFV, Xtest, ytest)
        utils.savefig("RBFKernelVal.png")
                    

    elif question == '4.1': 
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.jpg'.format(i))

    else:
        print("Unknown question: %s" % question)    