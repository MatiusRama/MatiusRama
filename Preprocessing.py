import pandas as pd

# Baca dataset
iris = pd.read_csv('iris.csv')

# Cek data teratas
print(iris.head())

# Cek tipe data kolom
print(iris.dtypes)

# Cek data hilang
print(iris.isna().sum())

# Drop kolom yang tidak dibutuhkan
iris.drop(['Id'], axis=1, inplace=True)

# Encoding kolom target
iris['Species'] = iris['Species'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Normalisasi data numerik
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])

# Pisahkan fitur dan target
X = iris.drop('Species', axis=1)
y = iris['Species']

# Split data menjadi data train dan test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

