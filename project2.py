
import json
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    with open('yummly.json', 'r') as f:
        data = json.load(f)
    return data


def preprocess_data(data):
    df = pd.DataFrame(data)
    df.drop_duplicates(subset='id', inplace=True)
    df.dropna(subset=['ingredients'], inplace=True)
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ','.join(x))
    return df


def train_classifier(df):
    vectorizer = CountVectorizer(lowercase=False)
    X_train = vectorizer.fit_transform(df['ingredients_str'])
    y_train = df['cuisine']
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return vectorizer, clf


def find_similar_recipes(args, vectorizer, clf, df):
    input_ingredients = args.ingredients
    input_ingredients_str = ','.join([ingr for sublist in input_ingredients for ingr in sublist])
    X_test = vectorizer.transform([input_ingredients_str])
    df['similarity'] = cosine_similarity(X_test, vectorizer.transform(df['ingredients_str']))[0]
    top_n_recipes = df.sort_values('similarity', ascending=False).head(args.N)
    top_n_recipes = [{"id": str(row["id"]), "score": round(row["similarity"], 2)} for _, row in top_n_recipes.iterrows()]
    return {
        'cuisine': clf.predict(X_test)[0],
        'score': round(df['similarity'].max(), 2),
        'closest': top_n_recipes
    }


def write_to_file( data):
    with open('../cs5293sp23-project21/output.json', 'w') as f:
        json.dump(data, f, indent=4)
    print(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the type of cuisine and find similar meals.')
    # parser.add_argument('--filename', metavar='F', type=str, default='yummly.json', help='name of input file')
    parser.add_argument('--ingredients', metavar='I', type=str, nargs='+', help='a list of ingredients', action='append')
    parser.add_argument('--N', metavar='N', type=int, default=5, help='number of similar meals to return')
    # parser.add_argument('--output', metavar='O', type=str, default='results.json', help='name of output file')
    args = parser.parse_args()

    data = load_data()
    df = preprocess_data(data)
    vectorizer, clf = train_classifier(df)
    result = find_similar_recipes(args, vectorizer, clf, df)
    write_to_file( result)
