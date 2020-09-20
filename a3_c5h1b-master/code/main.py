
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1.1
        # N,D = X.shape
        # items = np.zeros(D)
        # for n in range(len(item_ind)):
        #     items[item_ind[n]] += ratings["rating"][n]
        # print("Most Popular Item", item_inverse_mapper[np.argmax(items)])
        # print("Num Stars", max(items))

        # YOUR CODE HERE FOR Q1.1.2
        # array = np.bincount(user_ind)
        # person = np.argmax(array)
        # print("Person", user_inverse_mapper[person])
        # print("number of items", array[person])
        # YOUR CODE HERE FOR Q1.1.3
        
        # plt.hist(user_ind)
        # plt.yscale('log', nonposy = 'clip')
        # plt.title("Number of ratings per user")
        # plt.xlabel("")
        # plt.ylabel("")
        # fname = os.path.join("..", "figs", "q111.pdf")
        # plt.savefig(fname)

        # plt.hist(item_ind)
        # plt.yscale('log', nonposy = 'clip')
        # plt.title("Number of ratings per item")
        # plt.xlabel("")
        # plt.ylabel("")
        # fname = os.path.join("..", "figs", "q112.pdf")
        # plt.savefig(fname)

        # plt.hist(ratings["rating"])
        # plt.title("Ratings themselves")
        # plt.xlabel("")
        # plt.ylabel("")
        # fname = os.path.join("..", "figs", "q113.pdf")
        # plt.savefig(fname)

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]
        
        print(url_amazon % grill_brush)
        # YOUR CODE HERE FOR Q1.2
        nb = NearestNeighbors(n_neighbors=6)
        nb.fit(X.toarray().T)
        distances1, indices1 = nb.kneighbors(grill_brush_vec)
        print(ratings["item"][indices1])

        n = NearestNeighbors(n_neighbors=6, metric = "cosine")
        n.fit(X.toarray().T)
        distances1, indices3 = n.kneighbors(grill_brush_vec)
        print(ratings["item"][indices3])
        # YOUR CODE HERE FOR Q1.3
        for i in range(len(indices1)):
            print("Euclidean item rating:", np.sum(X[:,indices1[i]]))
            print("Cosine item rating:", np.sum(X[:,indices3[i]]))

    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # YOUR CODE HERE
        z = np.ones(500)
        z[400:] = 0.1
        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,z)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="31WLS.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares Bias, no bias",filename="41LSB.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            # Compute training error
            yhat = model.predict(X)
            trainError = np.mean((yhat - y)**2)
            print("Training error = %.1f" % trainError)

            # Compute test error
            if Xtest is not None and ytest is not None:
                yhat = model.predict(Xtest)
                testError = np.mean((yhat - ytest)**2)
                print("Test error     = %.1f" % testError)

    else:
        print("Unknown question: %s" % question)

