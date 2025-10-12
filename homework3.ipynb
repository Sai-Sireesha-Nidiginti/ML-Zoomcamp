import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
!curl -O https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
df=pd.read_csv('course_lead_scoring.csv')
df.info()
Data preparation
Check if the missing values are presented in the features.
If there are missing values:
For caterogiral features, replace them with 'NA'
For numerical features, replace with with 0.0
# Check if there are nulls in columns 
df.isnull().sum()
# get categorical and numerical columns
categorical = df.select_dtypes(include='object').columns.tolist()
numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f'Categorical columns: {categorical}')
print(f'Numerical columns: {numerical}')
# get a dataframe with only categorical columns
print(df[categorical].isnull().sum())

# get columns with nulls
cat_cols_with_nulls = df[categorical].columns[df[categorical].isnull().any()].tolist()
print(cat_cols_with_nulls)

# fill nulls in categorical columns with 'NA'
df[categorical] = df[categorical].fillna('NA')
# get a dataframe with only numerical columns
df[numerical].isnull().sum()

# get columns with nulls
numerical_cols_with_nulls = df[numerical].columns[df[numerical].isnull().any()].tolist()
print(numerical_cols_with_nulls)

# fill nulls in numerical columns with 0
df[numerical] = df[numerical].fillna(0)
Question 1
What is the most frequent observation (mode) for the column industry?

NA
technology
healthcare
retail
df['industry'].mode()
Question 2
Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features.

What are the two features that have the biggest correlation?

interaction_count and lead_score
number_of_courses_viewed and lead_score
number_of_courses_viewed and interaction_count
annual_income and interaction_count
Only consider the pairs above when answering this question.
df[numerical].corr()
sns.heatmap(df[numerical].corr(), cmap='coolwarm', annot=True)
plt.show()
Split the data
Split your data in train/val/test sets with 60%/20%/20% distribution.
Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.
Make sure that the target value y is not in your dataframe.
# create train, val, test splits
X_train,X_tmp,y_train,y_tmp=train_test_split(df.drop(columns=['converted']),df['converted'],test_size=0.4,random_state=42)
print(X_train.shape,y_train.shape,X_tmp.shape,y_tmp.shape)

X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
print(X_val.shape, y_val.shape, X_test.shape, y_test.shape)
Question 3
Calculate the mutual information score between y and other categorical variables in the dataset. Use the training set only.
Round the scores to 2 decimals using round(score, 2).
Which of these variables has the biggest mutual information score?

industry
location
lead_source
employment_status
df.info()
print(f'mutual_info_score industry: {round(mutual_info_score(df['industry'], df['converted']), 2)}')
print(f'mutual_info_score location: {round(mutual_info_score(df['location'], df['converted']), 2)}')
print(f'mutual_info_score lead_source: {round(mutual_info_score(df['lead_source'], df['converted']), 2)}')
print(f'mutual_info_score employment_status: {round(mutual_info_score(df['employment_status'], df['converted']), 2)}')
print(f"mutual_info_classif number_of_courses_viewed: {round(mutual_info_classif(df[['number_of_courses_viewed']], df['converted'])[0], 2)}")
print(f"mutual_info_classif annual_income: {round(mutual_info_classif(df[['annual_income']], df['converted'])[0], 2)}")
print(f"mutual_info_classif employment_status: {round(mutual_info_classif(df[['lead_score']], df['converted'])[0], 2)}")
print(f"mutual_info_classif interaction_count: {round(mutual_info_classif(df[['interaction_count']], df['converted'])[0], 2)}")
Question 4
Now let's train a logistic regression.
Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
Fit the model on the training dataset.
To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
Calculate the accuracy on the validation dataset and round it to 2 decimal digits.
What accuracy did you get?

0.64
0.74
0.84
0.94
# convert categorical columns to numeric using one-hot encoding
X_train = pd.get_dummies(X_train, columns=categorical, drop_first=True)
#X_train[categorical]
# convert categorical columns to numeric using one-hot encoding
X_val = pd.get_dummies(X_val, columns=categorical, drop_first=True)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
round(accuracy_score(y_val, model.predict(X_val)), 2)
Question 5
Let's find the least useful feature using the feature elimination technique.
Train a model using the same features and parameters as in Q4 (without rounding).
Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
Which of following feature has the smallest difference?

'industry'
'employment_status'
'lead_score'
Note: The difference doesn't have to be positive.
def recursive_feature_elimination(model, X_train, X_val, y_train, y_val, n_remove=1):
    """
    Iteratively removes the least important feature(s) (by absolute coefficient) from the model,
    retrains, and records the validation accuracy at each step.

    Args:
        model: Trained linear or logistic regression model.
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        n_remove (int): Number of features to remove at each step.

    Returns:
        dict: {removed_feature: accuracy_after_removal}
    """
    features = list(X_train.columns)
    results = {}
    current_X_train = X_train.copy()
    current_X_val = X_val.copy()
    current_model = model

    while len(features) > 1:
        # Find least important feature(s)
        abs_coefs = np.abs(current_model.coef_[0]) if hasattr(current_model.coef_[0], '__len__') else np.abs(current_model.coef_)
        idx = np.argsort(abs_coefs)[:n_remove]
        least_feats = [features[i] for i in idx]

        # Remove feature(s)
        features = [f for f in features if f not in least_feats]
        current_X_train = current_X_train[features]
        current_X_val = current_X_val[features]

        # Retrain model
        current_model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        current_model.fit(current_X_train, y_train)
        acc = accuracy_score(y_val, current_model.predict(current_X_val))

        for feat in least_feats:
            results[feat] = acc
            print(f"Removed: {feat}, Validation accuracy: {acc:.4f}")

        if len(features) <= n_remove:
            break

    return results

# Example usage:
# model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
# model.fit(X_train, y_train)
# results = recursive_feature_elimination(model, X_train, X_val, y_train, y_val, n_remove=1)
results = recursive_feature_elimination(model, X_train, X_val, y_train, y_val, n_remove=1)
# Subtract 0.74 from all values in the results dictionary to get the accuracy difference
accuracy_with_all_features = 0.74
accuracy_diff = {feat: acc - accuracy_with_all_features for feat, acc in results.items()}

# Sort accuracy_diff by value in ascending order
sorted_accuracy_diff = dict(sorted(accuracy_diff.items(), key=lambda item: item[1]))
print(sorted_accuracy_diff)
Question 6
Now let's train a regularized logistic regression.
Let's try the following values of the parameter C: [0.01, 0.1, 1, 10, 100].
Train models using all the features as in Q4.
Calculate the accuracy on the validation dataset and round it to 3 decimal digits.
Which of these C leads to the best accuracy on the validation set?

0.01
0.1
1
10
100
Note: If there are multiple options, select the smallest C.
C_values = [0.01, 0.1, 1, 10, 100]
val_accuracies = {}

for c in C_values:
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    val_accuracies[c] = round(acc, 3)
    print(f"C={c}: Validation accuracy = {round(acc, 3)}")

best_C = min([k for k, v in val_accuracies.items() if v == max(val_accuracies.values())])
print(f"Best C: {best_C} with accuracy {val_accuracies[best_C]}")
