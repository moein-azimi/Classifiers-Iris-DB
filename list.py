import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc



#Reading our csv file
df = pd.read_csv("Iris.csv")
class_names = df['variety'].unique()
#Labels : The last column
cols = list(df.columns.values)
X = df[cols[0:-1]]
Y = df[[cols[-1]]]
Y = Y.values.ravel()

#Randomly splitting the dataset into two sub-datasets: trainig , test . 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Array shape
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#Classifier?

x = input(" RandomForest\n SuportVectorMachine\n NaiveBayes\n LogisticRegression\n KNeighbors\n DecisionTreeClassifier\n ")

if x == 'RandomForest':
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    
elif x == 'SuportVectorMachine':
    from sklearn.svm import SVC
    model = svm.SVC()
    model.fit(X_train, Y_train)
 
elif x == 'NaiveBayes':
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, Y_train)
    
elif x == 'LogisticRegression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, Y_train)

elif x == 'KNeighbors':
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    
elif x == 'DecisionTreeClassifier':
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=2)
    model.fit(X_train, Y_train)
    
else: 
    y = input("Neural Network? Y : ") 
    if y == 'Y': 
        import tensorflow as tf
        print(tf.__version__)
        from tensorflow.python.keras.models import Sequential 
        from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
        from tensorflow.python.keras.optimizers import Adam
        from tensorflow.python.keras.callbacks import ModelCheckpoint
        Y_train = np.array(pd.get_dummies(Y_train))
        Y_test = np.array(pd.get_dummies(Y_test))
        model=Sequential()
        model.add(Dense(100,input_dim=4,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(50,activation='sigmoid'))        
        model.add(Dense(3,activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()
        num_epochs = 14
        num_batch_size = 4
        folder='saved_models'
        if os.path.isdir(folder):
            print("Exists")
        else:
            print("Doesn't exists")
            os.mkdir(folder)
        checkpointer = ModelCheckpoint(filepath='saved_models/1.hdf5', verbose=1, save_best_only=True)
        history = model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), callbacks=[checkpointer], verbose=1)

try:
    expected = Y_test
    predicted = model.predict(X_test)
    a = metrics.accuracy_score(expected, predicted)
    df0 = pd.DataFrame(metrics.confusion_matrix(expected, predicted), index=class_names, columns=class_names)
    figsize = (10,7)
    fontsize=14
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df0, cmap="YlGnBu", annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=20, ha='right', fontsize=fontsize)
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    plt.title('Confusion Matrix - accuracy: %s -%s'% (a , x))
    fig.savefig(f'figure+{x}.png')
    
except:
    Y_pred = model.predict(X_test)
    matrix = metrics.confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    f1score = metrics.f1_score(Y_test.argmax(axis=1), Y_pred.argmax(axis=1),average = 'macro')
    df1 = pd.DataFrame(matrix, index=class_names, columns=class_names)
    figsize = (10,7)
    fontsize=14
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df1, cmap="YlGnBu", annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=20, ha='right', fontsize=fontsize)
    #plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix - accuracy: %s '% (f1score))
    fig.savefig('confusion matrix.png')

