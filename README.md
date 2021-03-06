# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Introductions](#introductions)
5. [Licensing, Authors, Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code is tested using Python version 3.8.6. All necessary libraries are in requirements.txt file, run the following commands in project's root directory to install them.

`pip install -r requirements.txt`

## Project Motivation <a name="motivation"></a>

In this project, I applied what I learned in data engineering skills to analyze disaster data
from [Append](https://appen.com/) to build a model for an API that classifiers disaster messages. I built a machine learning pipeline to categorize disaster events given real messages so that these messages can be sent to an appropriate disaster relief agency. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualization of the data. Here are the steps to complete this project:
1. ETL step: Extract data from csv files, clean data, load data to SQLite database.
2. ML Pipeline: Tokenization, lemmatization, removing stop words. Build Machine Learning pipeline and find the best paramaters using Grid Search algorithm.
3. Flask app: Display results and visualizations.

## File Descriptions <a name="files"></a>

`- app`

`| - template`

`| |- master.html  # main page of web app`

`| |- go.html  # classification result page of web app`

`|- run.py  # Flask file that runs app`


`- data`

`|- disaster_categories.csv  # data to process` 

`|- disaster_messages.csv  # data to process`

`|- process_data.py  # extract, clean, and load data to database`

`|- DisasterResponse.db   # database to save clean data to`


`- models`

`|- train_classifier.py  # load data, train, evaluate, and save model`

`|- classifier.pkl  # saved model `


`- requirements.txt  # all necessary python packages`


`- README.md`

## Instructions <a name="introductions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
Thanks Udacity for providing the started code.
