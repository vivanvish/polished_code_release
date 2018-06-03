'''
    Script to show the comparison of the custom model's performance with that
    of scikit-learn's SVC.
'''
from sklearn.svm import SVC
from KernelSVM import KernelSVM
from sklearn.datasets import load_digits
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print('Loading the data and splitting into train and test sets.')
X, y = load_digits(return_X_y =True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2, random_state=42)


print('Preprocessing the data.')
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('......KernelSVM.......')
print('Training SVM with RBF Kernel with parameter lamda= 16.0 , sigma= 0.5 .')

clf = KernelSVM(lamda=16,kernel='rbf',sigma=0.5)
clf.mysvm(X_train,y_train)
preds = clf.predict(X_train,X_test)
print('Misclassification Error with lamda = 16, sigma = 0.5 : ',1-accuracy_score(y_test,preds))

print('......Sklearn SVC.......')
print('Training SVC with RBF Kernel with parameter C= 16.0 , gamma= auto .')

clf = SVC(C=16,kernel='rbf',gamma='auto')
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
print('Misclassification Error with C = 16, gamma = auto: ',1-accuracy_score(y_test,preds))