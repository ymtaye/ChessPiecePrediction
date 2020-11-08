from random import shuffle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
import numpy as np
from sklearn.neural_network import MLPClassifier
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

## Read Files here
Directory = 'Data\\Chessman-image-dataset\\Chess'
image_ID = 1

## OPTIONS
ratio = 0.8  ## Determines percentage of data for training set
color = 'n'
col = 'gray'
variances = [0.97]  ## Explained variance by
Models = ['ADA', 'RF', 'MLP', 'SVM']
# IDs-------0------1------2------3------4--
aspectSizes = [100, 400, 600]  ## Models take considerably long after the 600x600 size mark

## Indexes
ChessPiece = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']
##PieceLabels-----0--------1-------2---------3-------4--------5----

accTable = []
accTable.append(['Model', 'PCA', 'NonPCA', 'Components', 'size', 'Train:Test', 'varianceRetained'])

for explVar in variances:
    for size in aspectSizes:
        data_set = []
        for piece in ChessPiece:
            path = os.path.join(Directory, piece)
            classes_index = ChessPiece.index(piece)
            for image in os.listdir(path):
                try:
                    # data = cv2.imread(os.path.join(path, image), cv2.COLOR_BGR2RGB)
                    # new_data = cv2.resize(data, (size, size))

                    data = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                    new_data = cv2.resize(data, (size, size))
                    data_set.append([new_data, classes_index])
                except:
                    "Cant read image file"
        shuffle(data_set)  ##Shuffle dataset

        ## Splitting data into train/test using 80:20 ratio
        training_set = data_set[:int(ratio * len(data_set))]
        test_set = data_set[int(ratio * len(data_set)):]
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []

        for features, labels in training_set:
            train_X.append(features)
            train_Y.append(labels)
        train_Y = np.array(train_Y)

        for features, labels in test_set:
            test_X.append(features)
            test_Y.append(labels)
        test_Y = np.array(test_Y)

        train_X = np.array(train_X)
        test_X = np.array(test_X)

        if color == 'y':
            train_X = train_X.flatten().reshape(size)
            test_X = test_X.flatten().reshape(size)
            train_X = np.array(train_X).reshape((len(train_X), size * size * 3))
            test_X = np.array(test_X).reshape((len(test_X), size * size * 3))
        else:
            train_X = np.array(train_X).reshape((len(train_X), (size * size)))
            test_X = np.array(test_X).reshape((len(test_X), (size * size)))

        ## Feature reduction using principal component analysis
        pca_dims = PCA()
        ## Train PCA to determine components on training dataset
        #train_X = normalize(train_X)
        pca_dims.fit(train_X)
        cumsum = np.cumsum(pca_dims.explained_variance_ratio_)
        ## Component no that explains the detemined variance is selected
        comp = np.argmax(cumsum >= explVar) + 1  ## No of components
        pca = PCA(n_components=comp)
        ## Transform train_X using PCA
        train_X_transformed = pca.fit_transform(train_X)
        print(train_X_transformed.shape)
        ## Inverse PCA transformation back to 2d to display image
        train_X_inverse = pca.inverse_transform(train_X_transformed)

        for model in Models:

            if model == 'MLP':
                ## Fit multilayer Perceptron using PCA transformed data
                clf_pca = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state=1)
                clf_pca.fit(train_X_transformed, train_Y)

                ## Fit a seprarte model with simliar parameters  on untransformed data
                clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 20, 20), random_state=1,
                                    max_iter=400)
                clf.fit(train_X, train_Y)
            elif model == 'RF':
                ## Fit Random Forest using PCA transformed data
                clf_pca = RandomForestClassifier(n_estimators=30, max_depth=5)
                clf_pca.fit(train_X_transformed, train_Y)

                ## Fit a seprarte model with simliar parameters on untransformed data
                clf = RandomForestClassifier(n_estimators=30, max_depth=5)
                clf.fit(train_X, train_Y)
            elif model == 'ADA':
                ## Fit Adaptive boosting model using PCA transformed data
                base = RandomForestClassifier(n_estimators=30, max_depth=5)
                clf_pca = AdaBoostClassifier(n_estimators=30, base_estimator=base)

                ## Fit a seprarte model with simliar parameters on untransformed data
                clf = AdaBoostClassifier(n_estimators=30, base_estimator=base)
                clf_pca.fit(train_X_transformed, train_Y)
                clf.fit(train_X, train_Y)
            elif model == 'SVM':
                ## Fit Support Vector Machines using PCA transformed data
                clf_pca = svm.SVC(gamma=0.001, kernel='rbf')
                clf_pca.fit(train_X_transformed, train_Y)

                ## Fit a seprarte model with simliar parameters on untransformed data
                clf = svm.SVC(gamma=0.001, kernel='rbf')
                clf.fit(train_X, train_Y)

            ## Transform Test set to keep same cardinality with training set
            test_X_transformed = pca.transform(test_X)
            ## Predict using Model version which fitted on transformed data
            y_pred_pca = clf_pca.predict(test_X_transformed)
            ## Predict using Model which fit on original(un reduced) data
            y_pred = clf.predict(test_X)

            # print('Model: ' + model)
            ## Prediction Accuracy Stats
            accuracy_pca = accuracy_score(test_Y, y_pred_pca)
            accuracy = accuracy_score(test_Y, y_pred)
            ## Add statistics of run with set options
            runStat = [model, str(round(accuracy_pca, 2)), str(round(accuracy, 2)), str(comp), str(size), str(ratio),
                       str(explVar)]
            accTable.append(runStat)
            # print('One Set Done')

#print(accTable)

## Plot random image in dataset to visualize PCA reduction
## Note Saves random image from iteration using size and variance value
            '''
            f = plt.figure()
            f.add_subplot(1,2, 1)
            plt.title("Original Image")
            plt.imshow(train_X[0].reshape((size,size)), cmap=col)
            f.add_subplot(1,2, 2)

            plt.title("PCA image with "+str(explVar)+ "explainedVariance")
            plt.imshow(train_X_inverse[0].reshape((size,size)), cmap=col)
            plt.savefig('Image_'+ str(4)+'.png')
'''

