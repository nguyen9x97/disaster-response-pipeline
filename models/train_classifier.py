import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import re
import pickle


nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])


def load_data(database_filepath):
    """
    Load data from SQLite database
    PARAMS:
        - database_filepath: database name
    RETURNS:
        - X: Pandas Series -  independent features
        - Y: Pandas DataFrame - target variables
        - category_names: LIST - list of target names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Normalize, tokenize, lemmatize text, and remove stop words.
    PARAMS:
        - text: STRING
    RETURNS:
        - tokens: LIST - List of words after processing
    """
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build pipeline: vectorize, apply TF-IDF, classify using RandomForestClassifier.
    Use grid search to find better parameters.
    PARAMS: None
    RETURNS:
        - cv: Pipeline model with best parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    paramerters = {
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=paramerters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict test data, show the precision, recall and F1 score for each category name.
    PARAMS:
        - model: trained model
        - X_test: test data input
        - Y_test: true output
        - category_names: list of category names
    RETURNS: None
    """
    Y_pred = model.predict(X_test)
    for index, category_name in enumerate(category_names):
        report = classification_report(Y_test[category_name], Y_pred[:, index])
        print(f"#### Category name: {category_name} ####")
        print(report)


def save_model(model, model_filepath):
    """
    Save trained model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
