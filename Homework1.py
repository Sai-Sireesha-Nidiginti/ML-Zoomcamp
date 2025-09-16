pd.__version__
!curl -O https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
df_car=pd.read_csv('car_fuel_efficiency.csv')
#How many records are in the dataset?
 
df_car.__len__()
#How many fuel types are presented in the dataset?
df_car['fuel_type'].nunique()
# How many columns in the dataset have missing values?
df_car.isnull().any().sum()
# What's the maximum fuel efficiency of cars from Asia?
df_car[df_car['origin']=='Asia']['fuel_efficiency_mpg'].max()
# Find the median value of horsepower column in the dataset.
print("initial median:", df_car['horsepower'].median())
# Next, calculate the most frequent value of the same horsepower column.
print("most frequent:", df_car['horsepower'].mode())
# Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.
df_car['horsepower'].fillna(df_car['horsepower'].mode()[0], inplace=True)
#Now, calculate the median value of horsepower once again.
print("new median:", df_car['horsepower'].median())
# Select all the cars from Asia
df_car[df_car['origin']=='Asia']

#Select the first 7 values
df_car[df_car['origin']=='Asia'].head(7)
#Get the underlying NumPy array. Let's call it X.
X = df_car[df_car['origin']=='Asia'].head(7)[['vehicle_weight', 'model_year']].values
#Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
XTX = X.T.dot(X)
#Invert XTX.
XTX_inv = np.linalg.inv(XTX)
#Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
#Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = XTX_inv.dot(X.T).dot(y)
#What's the sum of all the elements of the result?
w.sum()
