# this programme will detect whether someone has anemia from his lab results with ML and Python

# import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import pickle

# Create Title and SubTitle
st.write("""
#Anemia Detector
detect anemia with ML and Python
""")

# Get the data
csv_file = pd.read_csv('Test_Anemia.csv')

# Set a Subheader
st.subheader('Data infomations : ')

# Show the data as a table
st.dataframe(csv_file)

# show statistics about the data
st.write(csv_file.describe())

# show the data as a chart

chart = st.bar_chart(csv_file)

# Split the data into independent 'X' and dependent 'Y' variables
X = csv_file.iloc[:, 0:5].values
Y = csv_file.iloc[:, -1].values
# Split the data set into 75% training data and 25% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# get the feature input from the user


def get_user_input():
    Hemoglobin = st.sidebar.slider('Hemoglobin', 0.0, 20.0, 10.0)
    MCH = st.sidebar.slider('MCH', 15.0, 37.0, 20.0)
    MCHC = st.sidebar.slider('MCHC', 23.0, 37.0, 28.0)
    MCV = st.sidebar.slider('MCV', 60.0, 120.0, 97.1)
    GENDER = st.sidebar.radio('Gender', ['Male(1)', 'Female(0)'])
    if GENDER == 'Male(1)':
        gender = 1
    elif GENDER == 'Female(0)':
        gender = 0

    # Store a dictionarry into a vairable
    user_data = {
        'Hemoglobin': Hemoglobin,
        'MCH': MCH,
        'MCHC': MCHC,
        'MCV': MCV,
        'GENDER': gender
    }
    # Transform the data into a Dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and diplaying the user inputs
st.subheader('User Input : ')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Save the model to AnemiaModel.pkl

filename = 'gpr_model.pkl'
pickle.dump(RandomForestClassifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))



# Show the model metrics
st.subheader('Model Test Accuracy Score : ')
st.write(str(accuracy_score(Y_test, loaded_model.predict(X_test)) * 100) + '%')

# Store the model predictions in a variable
prediction = loaded_model.predict(user_input)

# Set a subheader to predict for user data
st.subheader('Identification')
st.write(prediction)
