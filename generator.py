import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("table.csv")

# Preprocessing steps
# Location transformation
location_dict_new = {
    'Chennai': 7,
    'Noida': 6,
    'Bangalore': 5,
    'Hyderabad': 4,
    'Pune': 3,
    'Madurai': 2,
    'Lucknow': 1,
    'other place': 0,
}
data['New Location'] = data["Location"].apply(lambda x: location_dict_new.get(str(x), location_dict_new['other place']))

# One-hot encoding
gen = pd.get_dummies(data["Function"])
hr = pd.get_dummies(data["Hiring Source"])

# Marital Status adjustment
data['New Marital'] = data["Marital Status"].apply(
    lambda x: 'other status' if str(x) not in ['Single', 'Marr.'] else str(x))
Mr = pd.get_dummies(data["New Marital"])

# Promoted/Non Promoted conversion
data['New Promotion'] = data["Promoted/Non Promoted"].apply(lambda x: int(x == 'Promoted'))

# Emp. Group adjustment
Emp_dict_new = {
    'B1': 4,
    'B2': 3,
    'B3': 2,
    'other group': 1,
}
data['New EMP'] = data["Emp. Group"].apply(lambda x: Emp_dict_new.get(str(x), Emp_dict_new['other group']))
emp = pd.get_dummies(data["New EMP"])

# Job Role Match conversion
data['New Job Role Match'] = data["Job Role Match"].apply(lambda x: int(x == 'Yes'))

# Gender one-hot encoding
gend = pd.get_dummies(data["Gender "])
tengrp = pd.get_dummies(data["Tenure Grp."])

# Create final dataset
data = pd.concat([data, hr, Mr, emp, tengrp, gen, gend], axis=1)
data.drop(["table id", "name", "Marital Status", "Promoted/Non Promoted", "Function", "Emp. Group",
           "Job Role Match", "Location", "Hiring Source", "Gender ", 'Tenure', 'New Marital',
           'New EMP', 'Tenure Grp.', 'phone number'], axis=1, inplace=True)

# Convert all column names to strings
data.columns = data.columns.astype(str)

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

# Handle missing values for numeric columns
numeric_imputer = SimpleImputer(strategy="mean")
data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

# Handle missing values for categorical columns
categorical_imputer = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

# Model definitions
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

lr = LogisticRegression()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
svm = SVC()
rm = RandomForestClassifier()
gnb = GaussianNB()

# Splitting data and training
X = data.drop(columns=['Stay/Left'])
y = data['Stay/Left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

Random_Forest = RandomForestClassifier()
Random_Forest.fit(X_train, y_train)

# Save the trained model
joblib.dump(Random_Forest, "random_forest_model.pkl")
