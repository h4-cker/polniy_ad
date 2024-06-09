import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

# df = pd.read_csv('winemag-data_first150k.csv')
# df = pd.read_csv('cleaned_df.csv')
# df = pd.read_csv('new_column_df.csv')
df = pd.read_csv('new_z_df.csv')

# Очистка датасета
# df.drop(df.columns[0], axis=1, inplace=True)
# df.drop_duplicates(inplace=True)
# df['price'].fillna(df['price'].median(), inplace=True)
# df.fillna('Unknown', inplace=True)
# df.to_csv('cleaned_df.csv', index=False)

# категориальные - 1,2,3,6,7,8,9,10
# числовые - 4,5

# Добавление нового столбца
# df['price_for_points'] = df['price'] / df['points']
# df.to_csv('new_column_df.csv', index=False)

# Частотная таблица для country
# freq_table = df['country'].value_counts()
# print(freq_table)

# Среднее для всего датасета и выборки из 100
# mean_price = round(df['price'].mean(), 3)
# sample_df = df.sample(n=100)
# mean_price_sample = round(sample_df['price'].mean(), 3)
# print(mean_price, mean_price_sample)

# Описательные характеристики
# numeric_df = df.select_dtypes(include='number')
# mean_values = numeric_df.mean()
# median_values = numeric_df.median()
# mode_values = numeric_df.mode().iloc[0]
# max_values = numeric_df.max()
# min_values = numeric_df.min()
# var_values = numeric_df.var()
# std_values = numeric_df.std()

# print('Mean values:')
# print(mean_values)
# print('\nMedian values:')
# print(median_values)
# print('\nMode values:')
# print(mode_values)
# print('\nMax values:')
# print(max_values)
# print('\nMin values:')
# print(min_values)
# print('\nVariance values:')
# print(var_values)
# print('\nStandard deviation values:')
# print(std_values)

# Z-баллы для price
# price = df['price']
# mean_price = price.mean()
# std_price = price.std()
# z_scores = price.apply(lambda x: (x - mean_price) / std_price)
# df['z_scores'] = z_scores
# df.to_csv('new_z_df.csv', index=False)

# Корреляционная матрица
# sns.heatmap(df.corr(numeric_only=True).round(3), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")
# plt.show()

# Гистограмма для points
# plt.hist(df['points'], bins=10, edgecolor='black')
# plt.xlabel('points')
# plt.ylabel('Частота')
# plt.title('Гистограмма \'points\'')
# plt.show()

# Диаграмма размаха для price
# sns_plot = sns.boxplot(x='points', y='price', data=df)
# plt.title('Диаграмма размаха \'price\'')
# plt.show()

# Диаграмма рассеяния
# sns_plot = sns.scatterplot(y='points', x='price', hue='country', data=df)
# plt.ylabel('points')
# plt.xlabel('price')
# plt.show()

# Линейная регрессия
# X = df[['points']]
# y = df['price']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(r2_score(y_test, y_pred))

# Логистическая регрессия
X = df[['points', 'price']]
y = df['country'] == 'US'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))