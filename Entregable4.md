
# Entregable 4
## Modelos

### Regresión Lineal


```python
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
```


```python
mp=pd.read_csv('nmaxplanck.csv', index_col=0)
mp.shape
```




    (12107, 16)




```python
df=pd.DataFrame(mp)
```


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idpers</th>
      <th>idreg</th>
      <th>idarea</th>
      <th>año</th>
      <th>nota1</th>
      <th>nota2</th>
      <th>nota3</th>
      <th>nota4</th>
      <th>nota5</th>
      <th>nota6</th>
      <th>nota7</th>
      <th>Curso</th>
      <th>bimestre</th>
      <th>grado</th>
      <th>nivel</th>
      <th>Promedio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500</td>
      <td>1</td>
      <td>22</td>
      <td>2013</td>
      <td>11</td>
      <td>18</td>
      <td>10</td>
      <td>14</td>
      <td>14</td>
      <td>15</td>
      <td>15</td>
      <td>ALEM</td>
      <td>1</td>
      <td>1</td>
      <td>PRIMARIA</td>
      <td>13.857143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10500</td>
      <td>2</td>
      <td>12</td>
      <td>2013</td>
      <td>12</td>
      <td>17</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>15</td>
      <td>15</td>
      <td>ALEM</td>
      <td>1</td>
      <td>1</td>
      <td>PRIMARIA</td>
      <td>14.428571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10500</td>
      <td>3</td>
      <td>26</td>
      <td>2013</td>
      <td>16</td>
      <td>16</td>
      <td>17</td>
      <td>14</td>
      <td>14</td>
      <td>15</td>
      <td>15</td>
      <td>ALEM</td>
      <td>1</td>
      <td>1</td>
      <td>PRIMARIA</td>
      <td>15.285714</td>
    </tr>
  </tbody>
</table>
</div>




```python
d1 = df.idpers
d2 = df.Promedio
d3 = df.Curso

datos=pd.concat([d1, d2, d3], axis=1)
datos.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>idpers</th>
      <th>Promedio</th>
      <th>Curso</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500</td>
      <td>13.857143</td>
      <td>ALEM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10500</td>
      <td>14.428571</td>
      <td>ALEM</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_X = df.Promedio[:, np.newaxis]
df_y = df.idpers[:, np.newaxis]

df_X_train = df_X[:-20]
df_X_test = df_X[0:]


df_y_train = df_y[:-20]
df_y_test = df_y[0:] #df.Promedio[0:]

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(df_X_train, df_y_train)

# Make predictions using the testing set
df_y_pred = regr.predict(df_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df_y_test, df_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df_y_test, df_y_pred))

print('Precision model: ')
print(regr.score(df_X_train, df_y_train))

# Plot outputs
plt.scatter(df_X_test, df_y_test,  color='black')
plt.plot(df_X_test, df_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```

    Coefficients: 
     [[108.88029664]]
    Mean squared error: 5731914.76
    Variance score: 0.00
    Precision model: 
    0.003077606590285664



![png](output_7_1.png)


### Reglas de Asociación


```python
alumno = df['idpers'].astype('str')
Promedio = df['Promedio'].astype('str')
```


```python
Cursos = (df[df['año']==2013]
              .groupby(['idpers', 'Curso'])['Promedio']
              .sum().unstack().reset_index().fillna(0)
              .set_index('idpers'))
```


```python
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(Cursos > 0, min_support=0.06, use_colnames=True)
frequent_itemsets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>(ALEM)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1</td>
      <td>(COMP)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.5</td>
      <td>(MUSC)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5</td>
      <td>(PINT)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1</td>
      <td>(MUSC, ALEM)</td>
    </tr>
  </tbody>
</table>
</div>




```python
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(ALEM)</td>
      <td>(MUSC)</td>
      <td>0.1</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.05</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(MUSC)</td>
      <td>(ALEM)</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>2.0</td>
      <td>0.05</td>
      <td>1.125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(ALEM)</td>
      <td>(MUSC)</td>
      <td>0.1</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.05</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
</div>



## Exportación

#### CSV, EXCEL, JSON, XML


```python
# CSV
# mp=pd.read_csv('nmaxplanck.csv', index_col=0)
df.to_csv('exportacion.csv')
```


```python
# Excel
df.to_excel('exportacion.xlsx')
```


```python
# JSON
df.to_json('exportacion.json')
```


```python
# XML
df.to_html('exportacion.html')
```
