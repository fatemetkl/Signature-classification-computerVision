
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
class neuralnet:
    def __init__(self,layer):
        self.layer=layer

    def train(self,X_train,y_train):
        # Neural network
        model = Sequential()
        model.add(Dense(100, input_dim=33, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=100, batch_size=70)

        return model




    def test(self,model,X_test,y_test):
        y_pred = model.predict(X_test)
        # Converting predictions to label
        pred = list()
        for i in range(len(y_pred)):
            pred.append(np.argmax(y_pred[i]))
        # Converting one hot encoded test label to label
        test = list()
        for i in range(len(y_test)):
            test.append(np.argmax(y_test[i]))

        from sklearn.metrics import accuracy_score
        a = accuracy_score(pred, test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test, pred)
        print("############### neural network confusion matrix###################")
        print(cm)

        print("precision - recall - fscore : ",precision_recall_fscore_support(test, pred, average='macro'))

        return (a * 100)

