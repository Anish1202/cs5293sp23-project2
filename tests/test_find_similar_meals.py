import json
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pytest

from project2 import write_to_file, find_similar_recipes


# load data
@pytest.fixture(scope='session')
def data():
    with open('yummly.json', 'r') as f:
        data = json.load(f)
    return data

# preprocess data
@pytest.fixture(scope='session')
def df(data):
    df = pd.DataFrame(data)
    df.drop_duplicates(subset='id', inplace=True)
    df.dropna(subset=['ingredients'], inplace=True)
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ','.join(x))
    return df

# train classifier
@pytest.fixture(scope='session')
def clf(df):
    vectorizer = CountVectorizer(lowercase=False)
    X_train = vectorizer.fit_transform(df['ingredients_str'])
    y_train = df['cuisine']
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return vectorizer, clf

# test find_similar_recipes
def test_find_similar_recipes(df, clf):
    input_ingredients = [['chicken', 'rice']]
    args = argparse.Namespace(ingredients=input_ingredients, N=5)
    vectorizer, clf = clf
    result = find_similar_recipes(args, vectorizer, clf, df)
    assert len(result['closest']) == 5
    assert 'cuisine' in result.keys()
    assert 'score' in result.keys()
    assert 'closest' in result.keys()

# test write_to_file
def write_to_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(json.dumps(data))

def test_write_to_file(tmp_path):
    data = {'cuisine': 'italian', 'score': 0.5, 'closest': [{'id': '12345', 'score': 0.8}]}
    file_path = tmp_path / 'test.json'
    write_to_file(data, file_path)
    assert file_path.read_text() == json.dumps(data, indent=4)
