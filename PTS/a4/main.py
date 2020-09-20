import sys
import argparse
import linear_model
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400, verbose=1)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.logRegL2(lammy=1.0, maxEvals=400, verbose=1)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.2":
        # Load Binary and Multi -class data
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL1 model
        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400, verbose=1)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.3":
        # Load Binary and Multi -class data
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL0 model
        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.5":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        lrl2 = LogisticRegression(fit_intercept=False)
        lrl2.fit(XBin,yBin)
        print("\nTraining error %.3f" % utils.classification_error(lrl2.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(lrl2.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (lrl2.coef_ != 0).sum())

        lrl1 = LogisticRegression(penalty='l1',  fit_intercept=False)
        lrl1.fit(XBin,yBin)
        print("\nTraining error %.3f" % utils.classification_error(lrl1.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(lrl1.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (lrl1.coef_ != 0).sum())        


    elif question == "3":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Run Q3 given example - Fit One-vs-all Least Squares
        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMulti)))
        
    elif question == "3.2":
        # Load Binary and Multi -class data
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit One-vs-all Logistic Regression
        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    elif question == "3.3":
        # Load Binary and Multi -class data
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        ovr = LogisticRegression(C=1e8, fit_intercept=False)
        sft = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1e8, fit_intercept=False)

        ovr.fit(XMulti, yMulti)
        print("Training error %.3f" % utils.classification_error(ovr.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(ovr.predict(XMultiValid), yMultiValid))

        sft.fit(XMulti, yMulti)
        print("Training error %.3f" % utils.classification_error(sft.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(sft.predic