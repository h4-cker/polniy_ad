import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

df = pd.read_csv('diamonds.csv')

# Очистка датасета

df.drop(df.columns[0], axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv('cleaned_diamonds.csv', index=False)

# Средние значения для carat, depth, table, price, x, y, z
columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

for column in columns:
    mean_value = round(df[column].mean(), 5)
    median_value = df[column].median()
    mode_value = df[column].mode()[0]

    print(f'{column}(среднее) - {mean_value}')
    print(f'{column}(медиана) - {median_value}')
    print(f'{column}(мода) - {mode_value}')
    print()

# Гистограммы для колонок
columns = ['carat', 'depth', 'table']
for column in columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=50)
    plt.title(f'Гистограмма колонки \'{column}\'')
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.show()

# Boxplot для цены в зависимости от качества огранки и цвета алмазов
plt.figure(figsize=(10, 6))
sns_plot = sns.boxplot(x='cut', y='price', data=df)
plt.title('Зависимость цены от качества огранки')
plt.show()

plt.figure(figsize=(10, 6))
sns_plot = sns.boxplot(x='color', y='price', data=df)
plt.title('Зависимость цены от цвета')
plt.show()

# Scatterplot для изучения взаимосвязи между 'carat' и 'price'
plt.figure(figsize=(10, 6))
plt.scatter(df['carat'], df['price'])
plt.title('Scatter Plot of Carat vs Price')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()

# Cтандартное отклонение и диапазон для переменных 'carat', 'depth', и 'price'
carat_std = df['carat'].std()
depth_std = df['depth'].std()
table_std = df['table'].std()
price_std = df['price'].std()

carat_min = df['carat'].min()
carat_max = df['carat'].max()
carat_range = carat_max - carat_min

depth_min = df['depth'].min()
depth_max = df['depth'].max()
depth_range = depth_max - depth_min

table_min = df['table'].min()
table_max = df['table'].max()
table_range = depth_max - depth_min

price_min = df['price'].min()
price_max = df['price'].max()
price_range = price_max - price_min

print(f'Carat (стандартное отклонение): {carat_std}')
print(f'Depth (стандартное отклонение): {depth_std}')
print(f'Table (стандартное отклонение): {table_std}')
print(f'Price (стандартное отклонение): {price_std}')
print()
print(f'Carat (диапазон): {carat_range}')
print(f'Depth (диапазон): {depth_range}')
print(f'Table (диапазон): {table_range}')
print(f'Price (диапазон): {price_range}')

# Корреляционная матрица
sns.heatmap(df.corr(numeric_only=True).round(3), vmin=-1, vmax=+1, annot=True, cmap="coolwarm")
plt.show()