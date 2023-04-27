from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__)

# Define the route for the loan prediction form
@app.route('/')
def loan_prediction_form():
    return render_template('loan_prediction.html')

# Define the route for receiving form data and returning prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    rate_of_interest = float(request.form['rate_of_interest'])
    interest_rate_spread = float(request.form['interest_rate_spread'])
    upfront_charges = float(request.form['upfront_charges'])
    property_value = float(request.form['property_value'])
    ltv = float(request.form['ltv'])


    df = pd.read_csv(r"Loan_Default.csv")

    # df.info()

    # checking for the null values in the dataset
    # df.isnull().any()
    # there exists null values

    # dropping the year column as it do not works as a contributin feature for the classification
    df.drop(['year'],inplace=True,axis=1)

    # separating the numerical and categorical columns 
    numerical_col = df.select_dtypes(include=[np.number])
    categorical_col = df.select_dtypes(exclude=[np.number])


    # filling the categorical variables with the mode
    for i in categorical_col.columns:
        mode = categorical_col[i].mode()
        mode = mode[0]
        categorical_col[i].fillna(value=mode, inplace=True)


    lb = LabelEncoder()
    for i in categorical_col.columns:
        categorical_col[i] = lb.fit_transform(categorical_col[i])

    # filling the null values of numerical col with mean of that feature
    for i in numerical_col.columns:
        numerical_col[i] = numerical_col[i].fillna(numerical_col[i].mean())


    df_full = pd.concat([categorical_col, numerical_col], axis=1, join='inner')
    df_full.head()

    df = df_full[[c for c in df if c not in ['Status']] + ['Status']]
    print(df.head())


    # redefinning the X
    y = df['loan_limit']
    col = ['rate_of_interest','Interest_rate_spread', 'Upfront_charges','property_value','LTV']
    X = df[col]
    # X.shape


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)


    # print(accuracy_score(y_test,y_pred))
    # print(model.predict([]))

    test_data = [[rate_of_interest, interest_rate_spread, upfront_charges, property_value, ltv]]

    prediction = model.predict(test_data)[0]

    # if prediction==0:
    #     prediction = 'cf' 
    # else:
    #     prediction = 'ncf'
    return render_template('loan_prediction.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
