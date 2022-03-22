from datetime import datetime
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


metadata = pd.read_csv('Stock market data/stock_metadata.csv')

index = 0
for item in metadata["Company Name"]:
    print(f"{index} : " + item)
    index+=1

inp = int(input("\n\nSelect an index above: "))
print("\n\nTrining model.....\n\n")
data = pd.read_csv("Stock market data/"+metadata["Symbol"][inp] +".csv")
startdate = data["Date"][0]
index = 0
for i in data["Date"]:
    diff = datetime.strptime(i, "%Y-%m-%d") - datetime.strptime(startdate, "%Y-%m-%d")
    # print(diff.days) 
    data["Date"][index] = int(diff.days)
    index+=1
data = data.dropna(axis=1)
data = data.drop(["Symbol","Series"],axis=1)
# print(data)

X = np.array(data["Date"])
columns = data.drop(["Date"],axis=1).columns
Y = np.array(data.drop(["Date"],axis=1))

X = X.reshape(-1, 1)
# Y = Y.reshape(-1, 1)

# print(X.shape)
# print(Y.shape)

# print(columns)

Xtrain , Xtest , Ytrain , Ytest = train_test_split(X,Y,test_size=0.1)

model = LinearRegression()

model.fit(Xtrain,Ytrain)

acc = model.score(Xtest,Ytest)

# print(acc)

enter_a_date = input("\n\nEnter a date (YYYY-MM-DD): ")
diff = datetime.strptime(enter_a_date, "%Y-%m-%d") - datetime.strptime(startdate, "%Y-%m-%d")
pred = model.predict([[diff.days]])
print("\n")

index = 0
for prediction in pred[0]:
    print(columns[index] + " : " + str(prediction) + "\n")
    index += 1

print("\n\n")