import pandas as pd
import plotly.express as px
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
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
import seaborn as sns

#NASA
nasa= pd.read_csv('neo.csv')

dummies = pd.get_dummies(nasa["hazardous"])
dummies.head()

nasa.dropna(inplace = True)

nasa["outcome"] = dummies[True]

columns_to_drop = ["id", "hazardous", "name","orbiting_body","sentry_object", "miss_distance", "est_diameter_min", "est_diameter_max"]
nasa.drop(columns_to_drop, axis=1, inplace = True)

class_0 = nasa[nasa['outcome'] == 0]
class_1 = nasa[nasa['outcome'] == 1]

class_count_0, class_count_1 = nasa['outcome'].value_counts()
class_1_over = class_1.sample(class_count_0, replace=True)

nasa = pd.concat([class_1_over, class_0], axis=0)

target = nasa["outcome"]

input_columns = nasa.loc[:, nasa.columns != "outcome"]


x_train, x_test, y_train, y_test = train_test_split(input_columns, target, test_size=0.2)

#Decision tree
descisionTree = DecisionTreeClassifier(random_state=0)
descisionTree.fit(x_train, y_train)

#Function

class Asteroid:
    def __init__ (self, velocity, magnitude):
        self.relative_velocity = float(velocity)
        self.absolute_magnitude = float(magnitude)
        nasa = pd.read_csv('neo.csv')
        self.data = nasa
    def predict(self):
        d = {'relative_velocity': [self.relative_velocity], 'absolute_magnitude': [self.absolute_magnitude]}
        features = pd.DataFrame(data = d)
        pred = descisionTree.predict(features)
        new_list = list(pred)
        if (new_list == [0]):
            return False
        else:
            return True


