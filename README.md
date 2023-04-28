
# cs5293sp23-project2
Name: Anish Sunchu    OU ID: 113583802


## ****Description**:**
The goal of Project 2 is to create an application that takes a list of ingredients from a user and predicts the type of cuisine and suggests similar meals. Additionally, it should be able to help chefs who want to change their current meal without changing the ingredients.

The project requires training or indexing a food dataset provided in the yummly.json file. The user inputs a list of ingredients, and the model or search index predicts the cuisine type and suggests the top-N closest foods based on the input ingredients. The cuisine type serves as a label for each food set.

The main file of the project is project2.py, which accepts multiple --ingredient flags to add ingredients to the meal. If an ingredient has multiple words, quotes should be used around it. 

### **How to install:**


* -Create e new repository in github with name cs5293sp23-project2. Download the project file from git
* -create a directory in above project 'mkdir project1'
* -create one more module for tests 'mkdir tests'
* -add all the functional code in project1 and pytests in tests directory
* -now install pyenv using 'pip install pipenv'
* -install all required libraries
* -Now we are all set to run code
* -use command 'python project2.py --n 5 --ingredient paprika --ingredient banana --ingredient "rice krispies" '
* -' pipenv run python -m pytest ' to run tests

### **External Links :**
scikit

https://scikit-learn.org/stable/

Pytest

https://docs.pytest.org/en/7.2.x/

Pandas

https://pandas.pydata.org/

### **Functions:**
**1.load_data():** The load_data() method is a function that reads the yummly.json file and returns its contents in a Python dictionary format.

First, it opens the yummly.json file in read-only mode using the open() method, and the file handle is assigned to the variable f. Then, the json.load() method is used to parse the contents of the file into a dictionary object. Finally, the function returns the data dictionary object.


**2.preprocess_data():** The preprocess_data() function takes a JSON data object as input, converts it to a pandas DataFrame, removes duplicate rows based on the id column, drops any rows where the ingredients column is missing or null, and creates a new column called ingredients_str which concatenates the ingredients list into a comma-separated string. Finally, the function returns the cleaned DataFrame. This preprocessing step prepares the data for use in training a machine learning model or for searching for similar recipes based on ingredient input.

**3.train_classifier():** The train_classifier() function takes a preprocessed DataFrame as input, creates a CountVectorizer object to convert the ingredients strings into a sparse matrix of token counts, fits the vectorizer on the ingredients data, and transforms it into a matrix of features. It also creates a target variable y_train from the cuisine column in the DataFrame. The function then creates a MultinomialNB classifier object, fits it on the feature and target variables, and returns both the fitted vectorizer and classifier objects. This function trains a machine learning model that can be used to predict the cuisine of a dish based on its ingredients.

**4.find_similar_recipes():** This function takes in four arguments:

args: the command line arguments parsed by argparse that contains the list of input ingredients and the number of similar recipes to return.
vectorizer: the fitted CountVectorizer object used to transform the ingredients into a numerical representation.
clf: the trained MultinomialNB classifier object used to predict the cuisine of the input ingredients.
df: the preprocessed DataFrame that contains the recipe data.
The function first converts the input ingredients into a string format, and then transforms it using the same CountVectorizer object used to preprocess the training data. It then computes the cosine similarity between the input ingredients and all recipes in the DataFrame. The top N most similar recipes are then selected based on the cosine similarity score, and their IDs and similarity scores are returned as a list of dictionaries.

Finally, the predicted cuisine of the input ingredients is returned by using the trained classifier to predict the cuisine of the input ingredients based on their numerical representation. The highest similarity score among all recipes is also returned as a measure of how well the input ingredients match the recipes in the dataset. The returned output is a dictionary with the cuisine, similarity score, and list of closest recipes.

**5.write_to_file():** This function takes in a dictionary data and writes it to a file named output.json in a pretty-printed JSON format with an indentation of 4 spaces. It then prints the same JSON data to the console.


### **Test Functions:**
**1.data():** This function loads data from a JSON file named 'yummly.json' and returns it. Here's a step-by-step explanation of how it works:

* The with statement is used to open the file 'yummly.json' in read mode. This statement ensures that the file is closed properly after it has been read, even if an error occurs.
* The json.load(f) method is used to read the contents of the file and parse it as JSON. This method returns a Python object that represents the JSON data.
* The parsed JSON data is then assigned to the data variable.

Finally, the data variable is returned from the function.

**2.df():** The df() function converts the food data into a pandas DataFrame, cleans the data by removing duplicates and missing values, and adds a new column with comma-separated ingredients. This new column makes it easier to search for food items based on a specific set of ingredients. The cleaned DataFrame is then returned.

**3.clf():** The clf() method is a function that takes a pandas DataFrame object and returns a trained Naive Bayes classifier for predicting the type of cuisine based on the list of ingredients.

First, the CountVectorizer object is created with the parameter lowercase=False, which prevents the ingredient strings from being converted to lowercase. This is done to preserve any important casing information in the ingredient names.

Then, the fit_transform() method of the CountVectorizer object is called on the ingredients_str column of the DataFrame to convert each ingredient list into a matrix of token counts. The resulting matrix is assigned to X_train.

Next, the y_train variable is set to the cuisine column of the DataFrame, which contains the labels for the corresponding ingredient lists.

Finally, a MultinomialNB object is created as the classifier, and the fit() method is called with X_train and y_train as parameters to train the classifier. The trained vectorizer and classifier are then returned as a tuple.

4.**test_find_similar_recipes():** The test_find_similar_recipes function tests the find_similar_recipes function by passing input ingredients, a vectorizer, a classifier, and a Pandas DataFrame of recipe data. It then checks that the output dictionary from find_similar_recipes contains the expected information about the closest matching recipes.

5.**test_write_to_file():** The test_write_to_file function tests the write_to_file function by creating a temporary file path, creating a dictionary containing some recipe information, calling write_to_file with the dictionary and file path, and then checking that the contents of the file at the path match the expected JSON format of the dictionary. This ensures that the write_to_file function is correctly writing the dictionary to a JSON file.

### **Bugs:**

* In the find_similar_recipes function, the args.n parameter is accessed as args.N, which would cause an AttributeError.
* In the write_to_file function, the output file name is hardcoded as output.json, whereas the original code uses results.json. This could lead to confusion or unintentionally overwriting a file.

### **Assumptions:**

* The input JSON file has a specific format with the expected keys ('id', 'cuisine', 'ingredients', etc.). If the input file has a different format, the code would not work correctly.
* The input ingredients are assumed to be a list of strings, where each string represents an ingredient. If the input is given in a different format, such as a file or a database, additional code would be needed to process the input correctly.
* The cosine similarity metric is used to find similar recipes based on their ingredients. This assumes that the ingredients are the only important factor in determining similarity and does not take into account other factors such as cooking method, cuisine type, etc.
