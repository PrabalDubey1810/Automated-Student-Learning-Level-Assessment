import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('random_dataset.csv')
    return df

# Train the model
def train_model(df):
    vectorizer = TfidfVectorizer()
    classifier = SVC(kernel='linear')
    pipeline = Pipeline([('vectorizer', vectorizer), ('classifier', classifier)])

    X_train, X_test, y_train, y_test = train_test_split(df['question'], df['subject'], test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")

    return pipeline

# Main function
def main():
    st.title('Subject Focus Predictor')

    # Load data
    df = load_data()

    # Train the model
    model = train_model(df)

    # Input question from user
    question = st.text_input('Enter your question:')
    if question:
        # Predict subject and recommend topic
        predicted_subject = model.predict([question])[0]

        topics_mapping = {
            'physics': ['Newton\'s laws', 'kinematics', 'conservation laws'],
            'chemistry': ['stoichiometry', 'chemical bonding', 'kinetics'],
            'math': ['algebra', 'calculus', 'geometry']
        }

        recommended_topics = topics_mapping.get(predicted_subject, [])
        st.write(f"Predicted Subject: {predicted_subject}")
        st.write(f"Recommended Topics: {recommended_topics}")

if __name__ == '__main__':
    main()
