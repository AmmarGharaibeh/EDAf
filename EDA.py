import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
from prettytable import PrettyTable

df = pd.read_csv('D:\Birzet\MedicalCostPersonalDatasets.csv')

# print(df['region'].value_counts())
# print(df.info())
# print(df.describe())

# print(df.isnull().sum())

sns.boxplot(data=df[["charges"]])
# plt.show()

sns.set_style('whitegrid')
sns.displot(df['charges'], kde = False, color ='blue', bins = 30)
# plt.show()

df = pd.get_dummies(df)
# print(df.to_string())

df.drop(['sex_male','region_northeast','smoker_no'],axis=1,inplace=True)
# print(df.to_string())

X=df.drop('charges',axis=1)
Y=df.charges
X = sm.add_constant(X, prepend=True)
lm = sm.OLS(endog=Y, exog=X,)
lm = lm.fit()
# print(lm.summary())


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print('Size of the dataset = {}'.format(len(X)))
print('Size of the training dataset = {} ({}%)'.format(len(x_train), 100*len(x_train)/len(X)))
print('Size of the testing dataset = {} ({}%)'.format(len(x_test), 100*len(x_test)/len(X)))

lm = OLS(x_train,y_train)
result = lm.fit()
R2Score_train = lm.score(x_train, y_train)
R2Score_test = lm.score(x_test, y_test)



