import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv("wineQT.csv")

# delete ID Column
data.drop(['Id'], inplace = True, axis = 1)
data.info()

# Null 
data.isna().sum()

#the unique quality 

print("The Value Quality ",data["quality"].unique())

import plotly.express as px
fig1 = px.histogram(data,x='quality') #checking the distribution of quality variable
fig1.show()

fig,ax=plt.subplots(figsize=(15,15))
sns.countplot(x=data.quality).set_title('Target Distribution',size=15)


fig,ax=plt.subplots(6,2,figsize=(15,30))
sns.countplot(x=data.quality,ax=ax[0][0]).set_title('Target Distribution',size=15)
sns.boxplot(x=data.quality,y=data['volatile acidity'],ax=ax[0][1])
sns.boxplot(x=data.quality,y=data['citric acid'],ax=ax[1][0])
sns.boxplot(x=data.quality,y=data['residual sugar'],ax=ax[1][1])
sns.boxplot(x=data.quality,y=data['chlorides'],ax=ax[2][0])
sns.boxplot(x=data.quality,y=data['free sulfur dioxide'],ax=ax[2][1])
sns.boxplot(x=data.quality,y=data['total sulfur dioxide'],ax=ax[3][0])
sns.boxplot(x=data.quality,y=data['density'],ax=ax[3][1])
sns.boxplot(x=data.quality,y=data['pH'],ax=ax[4][0])
sns.boxplot(x=data.quality,y=data['sulphates'],ax=ax[4][1])
sns.boxplot(x=data.quality,y=data['alcohol'],ax=ax[5][0])
sns.boxplot(x=data.quality,y=data['fixed acidity'],ax=ax[5][1])

corr = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap="Blues", annot_kws={"fontsize":12})
plt.title("Correlation")

sns.pairplot(data,corner=True, hue='quality',
            x_vars=['density','alcohol','pH','volatile acidity','citric acid','sulphates','fixed acidity'],
            y_vars=['density','alcohol','pH','volatile acidity','citric acid','sulphates','fixed acidity']
            )

# if bigger than 6 best quality else lower quality

data["quality"] = [1 if i >= 6 else 0 for i in data.quality] 

print("Data Shape:", data.shape) 


corr_matrix = data.corr()

plt.show()

y = data.quality
x = data.drop(["quality"], axis = 1)

columns = x.columns.tolist()

sns.pairplot(data,corner=True, hue='quality',
            x_vars=['density','alcohol','pH','volatile acidity','citric acid','sulphates','fixed acidity'],
            y_vars=['density','alcohol','pH','volatile acidity','citric acid','sulphates','fixed acidity']
            )

fig,ax=plt.subplots(figsize=(15,15))
sns.countplot(x=data.quality).set_title('Target Distribution',size=15)



