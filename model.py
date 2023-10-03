import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

TRAINING_COLUMNS = model.feature_names_in_

# Preprocessing constants
gender_dict = {}
Marital_dict = {}
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
Emp_dict = {}


def preprocess_input(data):
    # Location
    data['New Location'] = data["Location"].apply(
        lambda x: location_dict_new.get(str(x), location_dict_new['other place']))

    # One-hot encoding
    gen = pd.get_dummies(data["Function"])
    hr = pd.get_dummies(data["Hiring Source"])

    # Marital Status
    data['New Marital'] = data["Marital Status"].apply(
        lambda x: 'other status' if str(x) not in Marital_dict else str(x))
    Mr = pd.get_dummies(data["New Marital"])

    # Promoted/Non Promoted
    data['New Promotion'] = data["Promoted/Non Promoted"].apply(lambda x: int(x == 'Promoted'))

    # Emp. Group
    def emp(x):
        return Emp_dict.get(str(x), 'other group')

    data['New EMP'] = data["Emp. Group"].apply(emp)
    emp = pd.get_dummies(data["New EMP"])

    # Job Role Match
    data['New Job Role Match'] = data["Job Role Match"].apply(lambda x: int(x == 'Yes'))

    # Gender
    data['New Gender'] = data["Gender "].apply(lambda x: gender_dict.get(x, 'other'))
    gend = pd.get_dummies(data["New Gender"])
    tengrp = pd.get_dummies(data["Tenure Grp."])

    # Create final dataset
    dataset = pd.concat([data, hr, Mr, emp, tengrp, gen, gend], axis=1)

    # Drop columns only if they exist
    columns_to_drop = ["table id", "name", "Marital Status", "Promoted/Non Promoted", "Function", "Emp. Group",
                       "Job Role Match", "Location", "Hiring Source", "Gender ", 'Tenure', 'New Gender',
                       'New Marital', 'New EMP', 'Tenure Grp.', 'phone number']
    dataset = dataset.drop(columns=dataset.columns.intersection(columns_to_drop))

    # Ensure consistency in columns
    for col in TRAINING_COLUMNS:
        if col not in dataset.columns:
            dataset[col] = 0  # add missing columns with all zeros

    dataset = dataset[TRAINING_COLUMNS]

    return dataset


def predict(data):
    # Preprocess the data
    processed_data = preprocess_input(data)

    # Use the trained model to make predictions
    prediction = model.predict(processed_data)

    return prediction
