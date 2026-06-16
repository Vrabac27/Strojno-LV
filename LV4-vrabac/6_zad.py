import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

df = pd.read_csv('cars_processed.csv')
df = df.drop(['name'], axis=1)
df_kodirani = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

X = df_kodirani.drop(['selling_price'], axis=1)
y = df_kodirani['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = lm.LinearRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print(mean_absolute_error(y_train, y_train_pred))
print(mean_squared_error(y_train, y_train_pred))
print(r2_score(y_train, y_train_pred))
print(max_error(y_train, y_train_pred))

print(mean_absolute_error(y_test, y_test_pred))
print(mean_squared_error(y_test, y_test_pred))
print(r2_score(y_test, y_test_pred))
print(max_error(y_test, y_test_pred))