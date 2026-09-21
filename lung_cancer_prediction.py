import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv("lung_cancer_data.csv")

df["GENDER"], _ = pd.factorize(df["GENDER"])

# Define features and target
x = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5,p=1,metric="minkowski")  # Hypertuning
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

#model accuracy 95.16%
