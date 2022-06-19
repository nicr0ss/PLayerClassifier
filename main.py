import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

players = pd.read_csv("Cleaned__5App_Players.csv")

X = players.drop(columns=["Position"])
y = players["Position"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train.values, y_train)

print("Welcome to the PLayer position predictor. \nI will predict the position of a player using data from the 2020 premier league season.")
player_input = np.array([])

for i in range(46):
  data = int(input("Please enter the value for " + players.columns[i] + ": "))
  player_input = np.append(player_input, data)

apps = int(input("Please enter how many appearances they made: "))
player_input = player_input / apps

try:
  predicted = knn.predict([player_input])
  print(predicted[0])
except ValueError:
  print("Array must be 2D and takes 46 arguments in season statistics.")