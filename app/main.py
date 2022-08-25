# import requirements needed
from flask import Flask, request, redirect, url_for, render_template, session
from utils import get_base_url
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from model import Asteroid

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12348
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url + 'static')


@app.route(f'{base_url}/results/')
def results():
    if 'data' in session:
        data = session['data']
        return render_template('input_model.html', generated=data)
    else:
        return render_template('input_model.html', generated=None)


# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')


# define additional routes here
# for example:
@app.route(f'{base_url}/input_model/', methods=["GET", "POST"])
def input_model():
    if request.method == 'POST':
        relative_velocity = request.form['relative_velocity']
        absolute_magnitude = request.form['absolute_magnitude']
        input_variables = Asteroid(relative_velocity, absolute_magnitude)
        result = input_variables.predict()
        ans = ''
        if result == 1:
            ans = "This asteroid is hazardous."
        else:
            ans = "This asteroid is non-hazardous."
        return render_template('input_model.html', var=ans)
    else:
        return render_template('input_model.html')


@app.route(f'{base_url}/store/')
def store():
    return render_template(
        'store.html')  # would need to actually make this page


@app.route(f'{base_url}/about/')
def about():
    return render_template(
        'about.html')  # would need to actually make this page


@app.route(f'{base_url}/products/')
def products():
    return render_template(
        'products.html')  # would need to actually make this page


if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc19.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host='0.0.0.0', port=port, debug=True)
