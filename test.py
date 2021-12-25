import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # normalization
from sklearn import metrics  # calculate score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from numpy import set_printoptions
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC  # support vector classifier, support vector regression
from sklearn.feature_selection import RFE, SelectKBest
import seaborn as sn  # heatmap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif

# Nir Maman - 313446841, May Moshe - 208832873

# from google.colab import files #upload the data file, that is going to be the input.
from google.colab import files
uploaded = files.upload()

data = np.genfromtxt('finaldata.csv', delimiter=',')
rowSize = len(data)
colSize = len(data[0])
y = np.ones((rowSize-1,1))
x = np.ones((rowSize-1, colSize-1))
for i in range(1, rowSize):
  for j in range(1, colSize): 
      x[i-1][j-1] = data[i][j]
  y[i-1] = data[i][0]
print(x)
std_scaler = StandardScaler()
std_scaler.fit(x)
std_data = std_scaler.transform(x)

# Connect of x and y (after normalization)
std_data = np.hstack((data[1:, 0:1], std_data))

# split to train test
data_frame_train, data_frame_test = train_test_split(std_data, test_size=0.1, random_state=42)

# Split test to x and y
x_test = data_frame_test.take(np.arange(1, colSize-1, 1), axis=1)
y_test_matrix = data_frame_test[:, 0:1]
y_test = np.zeros(len(y_test_matrix))
for i in range(len(y_test_matrix)):
    y_test[i] = y_test_matrix[i][0]
print(y_test)
# Split train to x and y + convert y from matrix to array
x_train = data_frame_train.take(np.arange(1, colSize-1, 1), axis=1)
vec_y = data_frame_train[:, 0:1]
y_train = np.zeros(len(vec_y))
for i in range(len(vec_y)):
    y_train[i] = vec_y[i][0]

print(y)


# Support vector machines (SVMs) are a set of supervised learning methods used for classification,
# regression and outliers detection.
def compare_knn_ovo_ovr_acc(x_train, y_train, x_test, y_test, accuracy_of_algorithms):
  # KNN:
  KNNAccuracy = []
  numOfNeighbors = [3, 4, 5] # need to change to: range(0,24) - without 9 or range(0,784)  
  for i in range(3):
    knn = KNeighborsClassifier(n_neighbors=numOfNeighbors[i])
    knn.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = knn.predict(x_test)
    KNNAccuracy.append(sklearn.metrics.accuracy_score(y_test, y_pred))
  accuracy_of_algorithms.append(np.max(KNNAccuracy))
  print("KNN\n" + str(np.max(KNNAccuracy)))
  plt.plot([3, 4, 5], KNNAccuracy, color='b')
  plt.title("KNN accuracies with different number of neighbors")
  plt.xlabel("Number of Neighbors")
  plt.ylabel("Accuracies")
  plt.show()

  # One VS One:
  ovo_clf = OneVsOneClassifier(SVC(class_weight='balanced'))
  ovo_clf.fit(x_train, y_train)
  y_predict = ovo_clf.predict(x_test)
  heatTableOVO = plt.axes()
  confusionMatOvO = confusion_matrix(y_test, y_predict)
  heatTableOVO = sn.heatmap(confusionMatOvO, annot=True, fmt='g', cmap="Greens")
  heatTableOVO.set_title('One VS One')
  plt.show()
  OvO_acc = sklearn.metrics.accuracy_score(y_test, y_predict)
  accuracy_of_algorithms.append(OvO_acc)
  print("One vs One accuracy:\n" + str(OvO_acc))

  # One VS Rest:
  ovr_clf = OneVsRestClassifier(SVC(class_weight='balanced'))
  ovr_clf.fit(x_train, y_train)
  y_predict = ovr_clf.predict(x_test)
  heatTableOVR = plt.axes()
  confusionMatOvR = confusion_matrix(y_test, y_predict)
  heatTableOVR = sn.heatmap(confusionMatOvR, annot=True, fmt='g', cmap="Greens")
  heatTableOVR.set_title('One VS Rest')
  plt.show()
  OvR_acc = sklearn.metrics.accuracy_score(y_test, y_pred)
  accuracy_of_algorithms.append(OvR_acc)
  print("One vs Rest accuracy:\n" + str(OvR_acc))
  
  
if __name__ == "__main__":
  accuracy_of_algorithms = []
  algorithms = ['KNN', 'OneVsOne', 'OneVsRest']
  compare_knn_ovo_ovr_acc(x_train, y_train, x_test, y_test, accuracy_of_algorithms)
  # Show compering one_vs_one, one_vs_all and KNN
  y_graph = np.arange(len(accuracy_of_algorithms))
  # Create bars and choose color
  plt.bar(y_graph, accuracy_of_algorithms, color='g')
  # Add title and axis names
  plt.title('Comparison of algorithms')
  plt.xlabel('Algorithms Types')
  plt.ylabel('Accuracies')
  # Create names
  plt.xticks(y_graph, algorithms)
  # Show graph
  plt.show() 




# Prepare models
models = []
models.append(('LR', LogisticRegression(solver='saga', tol=1e-3, max_iter=1000000)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVC', SVC(gamma='auto')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))

vector_y = y.flatten()
# Evaluate each model in turn in 4 measurements: accuracy, precision, recall and AUC

results = []
names = []
msgOverAll = []
for name, model in models:
	cv_results = model_selection.cross_val_score(model, std_data, vector_y, cv=10, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f" % (name, cv_results.mean())
	print(msg)
	msgOverAll.append(msg)
    
    
accur = []
for i in results:
  accur.append(i.mean())

y_graph = np.arange(len(accur))
# Create bars and choose color
plt.bar(y_graph, accur, color='b')
# Add title and axis names
plt.title('Algorithm Accuracy Comparison')
plt.xlabel('Algorithm Types')
plt.ylabel('Accuracies')
# Create names
plt.xticks(y_graph, names)
# Show graph
plt.show()
print(msgOverAll)