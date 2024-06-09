import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

train_df = pd.read_csv('TrainData_new.csv')
test_df = pd.read_csv('TestData_new.csv')

# Задание 1
# Проверьте, есть ли в тренировочных и тестовых данных пропуски? 
# Укажите количество столбцов тренировочной выборки, имеющих пропуски

print("\033[32m", train_df.isnull().sum())
print()
print(test_df.isnull().sum())
print("\033[0m")

# Задание 2
# a) В столбце с наибольшим количеством пропусков заполните пропуски средним значением по столбцу.
# В ответ запишите значение вычисленного среднего. Ответ округлите до десятых

f4_mean = train_df['feature_4'].mean()
train_df['feature_4'].fillna(f4_mean, inplace=True)

print("\033[32m", round(f4_mean, 1))
print("\033[0m")
# b) Найдите строки в тренировочных данных, где пропуски стоят в столбце с наименьшим количеством пропусков.
# Удалите эти строки. Сколько строк вы удалили?

train_df = train_df[~train_df["feature_6"].isna()]

# Задание 3
# a) Сколько столбцов в таблице (не считая target) содержат меньше 5 различных значений?

print("\033[32m", train_df.nunique())
print("\033[0m")

# b) Вычислите долю ушедших из компании клиентов, для которых значение признака 2 больше среднего значения по столбцу, 
# а значение признака 13 меньше медианы по столбцу. Ответ округлите до сотых

f2_mean = train_df['feature_2'].mean()
f13_median = train_df['feature_13'].median()

df_filtered = train_df[(train_df['feature_2'] > f2_mean) & (train_df['feature_13'] < f13_median)]

print("\033[32m", round(df_filtered['target'].mean(), 2))
print("\033[0m")

# Задание 4
# a) Разбейте тренировочные данные на целевой вектор y, содержащий значения из столбца target, и матрицу объект-признак X, содержащую остальные признаки.
# Обучите на этих данных логистическую регрессию из sklearn (LogisticRegression) с параметрами по умолчанию.
# Выведите среднее значение метрики f1-score алгоритма на кросс-валидации с тремя фолдами. Ответ округлите до сотых.

# При объявлении модели фиксируйте random_state = 42.
# Комментарий: параметры по умолчанию можете оставить дефолтными

print("\033[32m", train_df['cat_feature_1'].unique())
print(train_df['cat_feature_2'].unique())
print("\033[0m")

train_df['cat_feature_1'] = train_df['cat_feature_1'].map({"A": 0, "B": 1, "C": 2})
test_df['cat_feature_1'] = test_df['cat_feature_1'].map({"A": 0, "B": 1, "C": 2})

train_df['cat_feature_2'] = train_df['cat_feature_2'].map({"individuals": 0, "legal entities": 1})
test_df['cat_feature_2'] = test_df['cat_feature_2'].map({"individuals": 0, "legal entities": 1})

for col in train_df:
  train_df[col].fillna(train_df[col].median(), inplace=True)

for col in test_df:
  test_df[col].fillna(test_df[col].median(), inplace=True)

y = train_df['target']
X = train_df.drop('target', axis=1)

model = LogisticRegression(random_state=42)
model.fit(X, y)

f1_scores = cross_val_score(model, X, y, cv=3, scoring='f1')

print("\033[32m", round(f1_scores.mean(), 2))
print("\033[0m")

# Задание 5
# Подберите значение константы регуляризации C в логистической регрессии, перебирая гиперпараметр от 0.001 до 100 включительно, проходя по степеням 10.
# Для выбора C примените перебор по сетке по тренировочной выборке (GridSearchCV из библиотеки sklearn.model_selection)
# с тремя фолдами и метрикой качества - f1-score. Остальные параметры оставьте по умолчанию.
# В ответ запишите наилучшее среди искомых значение C.

# При объявлении модели фиксируйте random_state = 42.
# Комментарий: параметры по умолчанию можете оставить дефолтными

params = {'C': [0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(LogisticRegression(random_state=42), params, scoring='f1', cv=3)
grid_search.fit(X, y)
print("\033[32m", grid_search.best_params_)
print("\033[0m")

# b) Добавьте в тренировочные и тестовые данные новый признак 'NEW', равный произведению признаков '7' и '11'.
# На тренировочных данных с новым признаком заново с помощью GridSearchCV (с тремя фолдами и метрикой качества - f1-score)
# подберите оптимальное значение C (перебирайте те же значения C, что и в предыдущих заданиях),
# в ответ напишите наилучшее качество алгоритма (по метрике f1-score), ответ округлите до сотых.
# При объявлении модели фиксируйте random_state = 42.

train_df["NEW"] = train_df["feature_7"] * train_df["feature_11"]
test_df["NEW"] = test_df["feature_7"] * test_df["feature_11"]

y = train_df['target']
X = train_df.drop('target', axis=1)

params = {'C': [0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(LogisticRegression(random_state=42), params, scoring='f1', cv=3)
grid_search.fit(X, y)

print("\033[32m", grid_search.best_params_)
print(round(grid_search.best_score_, 2))
print("\033[0m")

# c) Теперь вы можете использовать любую модель машинного обучения для решения задачи.
# Также можете делать любую другую обработку признаков.
# Ваша задача - получить наилучшее качество по метрике F1-Score на тестовых данных.

# for col in train_df:
#   if train_df[col].dtype == 'object':
#     continue
#   train_df[col].fillna(train_df[col].median(), inplace=True)

# for col in test_df:
#   if test_df[col].dtype == 'object':
#     continue
#   test_df[col].fillna(test_df[col].median(), inplace=True)

y = train_df['target']
X = train_df.drop('target', axis=1)

model = CatBoostClassifier(cat_features=[14, 15])
model.fit(X, y)

pred = model.predict(test_df)

print("\033[32m", f1_score(test_df['target'], pred))
print("\033[0m")