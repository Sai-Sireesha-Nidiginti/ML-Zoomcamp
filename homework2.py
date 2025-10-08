import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df_car=pd.read_csv('car_fuel_efficiency.csv')
df_car.info()
# Preparing the dataset: Use only the following columns:

###### 'engine_displacement',
###### 'horsepower',
###### 'vehicle_weight',
###### 'model_year',
###### 'fuel_efficiency_mpg'
df_pred=df_car[['engine_displacement','horsepower','vehicle_weight','model_year','fuel_efficiency_mpg']]
# EDA
df_pred.head()
df_pred.info()
## Look at the fuel_efficiency_mpg variable. Does it have a long tail?
#Look at the fuel_efficiency_mpg variable. Does it have a long tail?
plt.figure(figsize=(10,6))
sns.histplot(df_pred['fuel_efficiency_mpg'], bins=30)
plt.title('Distribution of Fuel Efficiency (MPG)')
# Is mean>median?
df_pred['fuel_efficiency_mpg'].describe()
# Is skewness>0.5
df_pred['fuel_efficiency_mpg'].skew()
#is kurtosis>3
df_pred['fuel_efficiency_mpg'].kurtosis()
#### Based on the values and the histogram:

###### The mean (14.985) is slightly less than the median (15.006), which usually suggests a left (negative) skew.
###### The skewness is -0.01206, which is very close to 0, indicating the distribution is almost symmetric (not significantly skewed).
###### The kurtosis is 0.0227, which is close to 0 (normal distribution kurtosis is 0 in pandas), so the tails are similar to a normal distribution.
###### The histogram appears roughly symmetric and bell-shaped.
#### Conclusion:
###### The distribution of fuel_efficiency_mpg is approximately symmetric and not significantly skewed. The small negative skewness and the mean being just below the median suggest a very slight left skew, but it is negligible.
## Question 1
### There's one column with missing values. What is it?

df_pred.columns[df_pred.isnull().sum()>0]
### horsepower is the column that has missing values
## Question 2
### What's the median (50% percentile) for variable 'horsepower'?

df_pred['horsepower'].describe()
df_pred['horsepower'].median()
df_pred['horsepower'].quantile(0.5)
### horsepower has median of 149
### Prepare and split the dataset
### Shuffle the dataset (the filtered one you created above), use seed 42.
### Split your data in train/val/test sets, with 60%/20%/20% distribution.
# Shuffle the dataset and split into train/val/test sets

# Shuffle the dataset
df_shuffled = df_pred.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute split sizes
n = len(df_shuffled)
n_train = int(0.6 * n)
n_val = int(0.2 * n)
n_test = n - n_train - n_val

# Split the data
df_train = df_shuffled.iloc[:n_train]
df_val = df_shuffled.iloc[n_train:n_train+n_val]
df_test = df_shuffled.iloc[n_train+n_val:]

# Check the sizes
print(len(df_train), len(df_val), len(df_test))
from sklearn.model_selection import train_test_split

# First split: 60% train, 40% temp
df_train, df_temp = train_test_split(df_pred, test_size=0.4, random_state=42, shuffle=True)

# Second split: 20% val, 20% test from temp (which is 40% of data)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

# Check the sizes
print(len(df_train), len(df_val), len(df_test))
Question 3
We need to deal with missing values for the column from Q1.
We have two options: fill it with 0 or with the mean of this variable.
Try both options. For each, train a linear regression model without regularization using the code from the lessons.
For computing the mean, use the training only!
Use the validation dataset to evaluate the models and compare the RMSE of each option.
Round the RMSE scores to 2 decimal digits using round(score, 2)
Which option gives better RMSE?
Options:

With 0
With mean
Both are equally good
# replace missing values in horsepower with 0
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)
df_train_zero = df_train.copy()
df_train_zero['horsepower'].fillna(0, inplace=True)
#df_train_zero.replace({'horsepower': 0}, inplace=True)
df_train_zero['horsepower'].info()
df_val_zero = df_val.copy()
df_val_zero['horsepower'].fillna(0, inplace=True)

df_test_zero = df_test.copy()
df_test_zero['horsepower'].fillna(0, inplace=True)
X_train = df_train_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_train = df_train_zero['fuel_efficiency_mpg'].values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)
rmse(y_train, y_pred)
X_val = df_val_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_val = df_val_zero['fuel_efficiency_mpg'].values
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
val_zero_rmse = round(rmse(y_val, y_pred), 2)
# replace missing values in horsepower with mean
df_train_mean = df_train.copy()
df_train_mean['horsepower'].fillna(df_train['horsepower'].mean(), inplace=True)
df_train_mean.info()
df_val_mean = df_val.copy()
df_val_mean['horsepower'].fillna(df_train['horsepower'].mean(), inplace=True)

df_test_mean = df_test.copy()
df_test_mean['horsepower'].fillna(df_train['horsepower'].mean(), inplace=True)
X_train = df_train_mean[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_train = df_train_mean['fuel_efficiency_mpg'].values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)
rmse(y_train, y_pred)
X_val = df_val_mean[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_val = df_val_mean['fuel_efficiency_mpg'].values
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

val_mean_rmse = round(rmse(y_val, y_pred), 2)
Now let's train a regularized linear regression.
For this question, fill the NAs with 0.
Try different values of r from this list: [0, 0.01, 0.1, 1, 5, 10, 100].
Use RMSE to evaluate the model on the validation dataset.
Round the RMSE scores to 2 decimal digits.
Which r gives the best RMSE?
If multiple options give the same best RMSE, select the smallest r.
X_train = df_train_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_train = df_train_zero['fuel_efficiency_mpg'].values

w0, w = train_linear_regression_reg(X_train, y_train)

y_pred = w0 + X_train.dot(w)
rmse(y_train, y_pred)
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    X_train = df_train_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
    y_train = df_train_zero['fuel_efficiency_mpg'].values
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = df_val_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
    y_val = df_val_zero['fuel_efficiency_mpg'].values
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)

    print(f'for r of {r} the intercept is {w0} and the score is {round(score, 2)}')
Question 5
We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
For each seed, do the train/validation/test split with 60%/20%/20% distribution.
Fill the missing values with 0 and train a model without regularization.
For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
Round the result to 3 decimal digits (round(std, 3))
What's the value of std?

0.001
0.006
0.060
0.600
Note: Standard deviation shows how different the values are. If it's low, then all values are approximately the same. If it's high, the values are different. If standard deviation of scores is low, then our model is stable.
rmse_scores = []

for seed in range(10):
    # Split data
    df_train, df_temp = train_test_split(df_pred, test_size=0.4, random_state=seed, shuffle=True)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed, shuffle=True)

    # Fill missing values with 0
    df_train_zero = df_train.copy()
    df_val_zero = df_val.copy()
    df_train_zero['horsepower'].fillna(0, inplace=True)
    df_val_zero['horsepower'].fillna(0, inplace=True)

    # Prepare features and target
    X_train = df_train_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
    y_train = df_train_zero['fuel_efficiency_mpg'].values
    X_val = df_val_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
    y_val = df_val_zero['fuel_efficiency_mpg'].values

    # Train model
    w0, w = train_linear_regression(X_train, y_train)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    rmse_scores.append(score)
# Calculate standard deviation
std = np.std(rmse_scores)
print('Standard deviation:', round(std, 3))
Question 6
Split the dataset like previously, use seed 9.
Combine train and validation datasets.
Fill the missing values with 0 and train a model with r=0.001.
What's the RMSE on the test dataset?
Options:

0.15
0.515
5.15
51.5
# Split the data with seed 9
df_train, df_temp = train_test_split(df_pred, test_size=0.4, random_state=9, shuffle=True)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=9, shuffle=True)

# Combine train and validation sets
df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)

# Fill missing values with 0
df_full_train_zero = df_full_train.copy()
df_full_train_zero['horsepower'].fillna(0, inplace=True)
df_test_zero = df_test.copy()
df_test_zero['horsepower'].fillna(0, inplace=True)

# Prepare features and target
X_full_train = df_full_train_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_full_train = df_full_train_zero['fuel_efficiency_mpg'].values
X_test = df_test_zero[['engine_displacement','horsepower','vehicle_weight','model_year']].values
y_test = df_test_zero['fuel_efficiency_mpg'].values

# Train regularized model with r=0.001
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
y_pred = w0 + X_test.dot(w)
test_rmse = rmse(y_test, y_pred)
print('Test RMSE:', round(test_rmse, 3))

