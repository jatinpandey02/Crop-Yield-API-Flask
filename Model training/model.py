import numpy as np 
import pandas as pd 
import sklearn
print(sklearn.__version__)
df = pd.read_csv('final.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
#df = pd.get_dummies(df, columns=["Item"], prefix = ["Item"])
features=df[df.Area == 'India'].loc[:, df.columns != 'hg/ha_yield']
#features=df.loc[:, df.columns != 'hg/ha_yield']
label=df[df.Area == 'India' ]['hg/ha_yield']
#label=df['hg/ha_yield']
features.head()
features.drop(['Year','Area'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
enc = LabelEncoder()
features['Item'] = enc.fit_transform(features.Item)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(features, label, test_size=0.4, random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
clf=RandomForestRegressor()
model=clf.fit(train_x,train_y)

print(mean_squared_error(train_y,model.predict(train_x))/len(train_x))
print(mean_squared_error(test_y,model.predict(test_x))/len(test_x))
print(r2_score(train_y,model.predict(train_x)))
print(r2_score(test_y,model.predict(test_x)))
from pickle import dump

dump(model,open('model.pkl','wb'))
dump(enc,open('label.pkl','wb'))
from pickle import load

model = load(open('model.pkl','rb'))
enc = load(open('label.pkl','rb'))
input = [enc.transform(['Cassava'])[0],1083.0,75000.0,16.23]
input = np.array([input])
print(model.predict(input))
