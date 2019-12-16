
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
class svm():

    def train_predict_svm(self,feature_np,label_np):
        X_train, X_test, y_train, y_test = train_test_split(feature_np, label_np, test_size=0.1)  # 10%train

        model = LinearSVC()
        model.fit(X_train, y_train)

        prediction = []
        for x in X_test:
            prediction.append(model.predict(x.reshape(1, -1)))

        acc = 0
        for i in range(0, len(prediction)):
            if prediction[i] == y_test[i]:
                acc += 1

        pred = [int(i) for i in prediction]
        test = [int(i) for i in y_test]

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test, pred)
        print("############### SVM confusion matrix###################")
        print(cm)

        print("precision - recall - fscore : ",precision_recall_fscore_support(test, pred, average='macro'))
        acc=acc / len(prediction)
        return acc