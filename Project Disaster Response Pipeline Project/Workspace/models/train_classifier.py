import sys
import re
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import warnings
import pickle

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
stop_words = stopwords.words("english")
warnings.filterwarnings("ignore")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    '''
    load_data
    Load data from a sqlite database
    
    inputs
    database_filepath: filepath to the sqlite database file

    Returns:

    X: returns the messsages data
    Y: returns all the targets data
    targets: returns all the category names from targets

    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('tb_messages', engine)
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    targets = Y.columns.values
    
    return X, Y, targets
  

def tokenize(text):
    '''
    tokenize
    This function transform the text removing punctuation and normalize case, after lemmatize and removes stop words
    
    inputs
    text: message text

    Returns:

    tokens: returns the text's tokens 

    '''    
    # Normalize case and remove punctuation
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', string = text)
    # Tokenize text
    tokens = nltk.tokenize.word_tokenize(text)
    # Lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return tokens

def build_model():
    '''
    build_model
    This function creates a Pipeline that applies count vectorizer and TFIDF, after creates a Naive Bayes model to each text category and optimize their hyperparameters with grid search
    
    Returns:

    cv: Grid Search object ready to fit the model

    '''        
    pipeline = Pipeline([
        ('count_vec', CountVectorizer(tokenizer = tokenize)),
        ('tfdf', TfidfTransformer()),
        ('nb', MultiOutputClassifier(MultinomialNB()))
    ])
    parameters = {
        'nb__estimator__alpha': [1.0, 1.5], 
    }
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    This function evaluates the model's performance
    Inputs:

    model: estimator
    X_test: messages test data
    Y_test: category test data 
    category_names: category names
    

    Returns:

    text: performance metrics for each category model

    '''      
    y_pred = model.predict(X_test)
    for i in range(36):
        print(f"{category_names[i]} category:")
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    save_model
    This function saves the model to a pickle file
    Inputs:

    model: estimator
    model_filepath: the path you want to save the pickle file
    

    Returns:

    saves a pickle file to the directory informed in the model_filepath

    '''    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()