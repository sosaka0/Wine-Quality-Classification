from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("wineQT.csv")

# ID sütununu siliyoruz
data.drop(['Id'], inplace = True, axis = 1)
data.info()

# Null değer
data.isna().sum()

#data.info() 

describe = data.describe()

corr_matrix = data.corr()

data["quality"] = [1 if i >= 6 else 0 for i in data.quality] 

y = data.quality
x = data.drop(["quality"], axis = 1)


columns = x.columns.tolist()


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size =0.25,random_state = 41)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

X_train_df = pd.DataFrame(X_train, columns = columns)
X_train_df_describe = X_train_df.describe()
X_train_df["target"] = Y_train

accuracy_all = []

kNN = KNeighborsClassifier()
kNN = kNN.fit(X_train,Y_train)
y_dash = kNN.predict(X_test)
#accuracy_score(Y_test, y_dash)

accuracy_all.append(accuracy_score(y_dash, Y_test))


y_pred_test = kNN.predict(X_test)
y_pred_train = kNN.predict(X_train)

print("KNN Accuracy: {0:.2%}".format(accuracy_score(y_dash, Y_test)))
cm_test = confusion_matrix(Y_test, y_pred_test)
cm_train = confusion_matrix(Y_train, y_pred_train)
    
acc_test = accuracy_score(Y_test, y_pred_test) 
acc_train = accuracy_score(Y_train, y_pred_train)
print("Test Score: {}, Train Score: {}".format(acc_test, acc_train))
print()
print("CM Test: ",cm_test)
print("CM Train: ",cm_train)
print(classification_report(Y_test,y_pred_test))

con=(confusion_matrix(Y_test,y_pred_test))
print(con)
classes = [0, 1]
# plot confusion matrix
plt.imshow(con, interpolation='nearest', cmap=plt.cm.Reds)
plt.title("knn")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    plt.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.legend()
plt.show()
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_test)
plt.plot(fpr, tpr, marker='.', label='KNN')

plt.show()





