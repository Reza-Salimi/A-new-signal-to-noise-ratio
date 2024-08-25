import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
import scikitplot as skplt
%matplotlib inline

def process_data():
    columns_to_named = ['Adj']
    ind = pd.read_excel(r"C:\Users\123\Downloads\mysql\SPY.csv",parse_dates=True)
    ind_norm = (ind['Adj']-ind['Adj'].mean())/ind['Adj'].std(ddof=0)
    x_data = ind_norm.to_frame()
    y = ind['Adj Close']
    kind=[]
    b=y.median()
    for i in y:
        if i>=b:
            kind.append(1)
        if i<b:
            kind.append(0)
    y = pd.DataFrame({'P':kind})
    y_data = y['P']
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 101)
    return x_train, x_test, y_train, y_test

 # Define Feature Map

  def create_feature_column():
    feat_Adj = tf.feature_column.numeric_column('Adj')
    feature_column = [feat_Adj]
    return feature_column

x_train, x_test, y_train, y_test = process_data()
feature_column = create_feature_column()

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x_train, y_train, batch_size=50, num_epochs= 1000, shuffle=True)
#.numpy_input_fn(x_train, y_train, batch_size=50, num_epochs= 1000, shuffle=True)

eval_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x_test, y_test, batch_size=50, num_epochs= 1, shuffle=False)
#.numpy_input_fn(x_test, y_test, batch_size=50, num_epochs= 1, shuffle=False)

predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, num_epochs=1, shuffle=False)
#.inputs.numpy_input_fn(x=x_test, num_epochs=1, shuffle=False)

# Create Model

dnnmodel = tf.estimator.DNNClassifier(hidden_units=[20,20],
                                      feature_columns=feature_column,
                                      n_classes=2,
                                      activation_fn=tf.nn.softmax,
                                      dropout= None,
                                      optimizer=tf.optimizers.Adam(learning_rate=0.01)
                                     )

# Train, Test, and Evaluate

history = dnnmodel.train(input_fn=input_func,steps=500
                       )
dnnmodel.evaluate(eval_func)
