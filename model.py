
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("wineQT.csv")

# delete ID Column
data.drop(['Id'], inplace = True, axis = 1)


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

ETC = ExtraTreesClassifier()
ETC = ETC.fit(X_train,Y_train)

ETC_pred = ETC.predict(X_test) #Predictions on Testing data
print(ETC_pred)

# create model

import joblib

filename = 'model.joblib'
joblib.dump(ETC, filename)


 
# some time later...
 
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)