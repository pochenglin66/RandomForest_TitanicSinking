from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, \
    recall_score, classification_report, \
    accuracy_score, f1_score
import pandas as pd

titanic_test = pd.read_csv("./data/titanic_sinking/test.csv")
titanic_train = pd.read_csv("./data/titanic_sinking/train.csv")

# preprocessing
# training set factorization
Sex = pd.factorize(titanic_train['Sex'])[0]
titanic_train['Sex'] = Sex
Embarked = pd.factorize(titanic_train['Embarked'])[0]
titanic_train['Embarked'] = Embarked
Cabin = pd.factorize(titanic_train['Cabin'])[0]
titanic_train['Cabin'] = Cabin

# testing set factorization
Sex = pd.factorize(titanic_test['Sex'])[0]
titanic_test['Sex'] = Sex
Embarked = pd.factorize(titanic_test['Embarked'])[0]
titanic_test['Embarked'] = Embarked
Cabin = pd.factorize(titanic_test['Cabin'])[0]
titanic_test['Cabin'] = Cabin

# testing set missing value filling
titanic_test['Age'].fillna(value=0, inplace=True)

# training set missing value filling
titanic_train['Age'].fillna(value=0, inplace=True)
print(titanic_train.head())
#print(titanic_train.isnull().sum())

# I delete the name & ticket columns
tfeatures= titanic_test.columns[1:8]
features = titanic_train.columns[2:9]
#print(titanic_train[features].head())

y = titanic_train['Survived']

# Training
# Create a random forest classifier
clf = RandomForestClassifier(n_jobs=2)

# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(titanic_train[features], y)

# Apply the classifier we trained to the test data (which, remember, it has never seen before)
prediction = clf.predict(titanic_train[features])

#evaluation
print ('Accuracy:', accuracy_score(y, prediction))
print ('F1 score:', f1_score(y, prediction))
print ('Recall:', recall_score(y, prediction))
print ('Precision:', precision_score(y, prediction))
print ('\n classification report:\n', classification_report(y,prediction))

test_predict = clf.predict(titanic_test[tfeatures])
PassengerId = titanic_test['PassengerId']

result = pd.DataFrame(PassengerId, columns=['PassengerId'])
result['Survived'] = test_predict
result.to_csv("./data/titanic_sinking/test_answer.csv", index=False)
