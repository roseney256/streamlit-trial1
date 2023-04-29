import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from math import sqrt
import plotly.express as px
from PIL import Image
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

sns.set_style("darkgrid")
style.use('ggplot')

import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from math import sqrt
import plotly.express as px
from PIL import Image
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np
sns.set_style("darkgrid")


st.title("SIRGE APP")

col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

# st.markdown("""
# 	The data set contains information about money spent on advertisement and their generated sales. Money
# 	was spent on TV, radio and newspaper ads.
# 	## Problem Statement
# 	Sales (in thousands of units) for a particular product as a function of advertising budgets (in thousands of
# 	dollars) for TV, radio, and newspaper media. Suppose that in our role as statistical consultants we are
# 	asked to suggest.
# 	Here are a few important questions that you might seek to address:
# 	- Is there a relationship between advertising budget and sales?
# 	- How strong is the relationship between the advertising budget and sales?
# 	- Which media contribute to sales?
# 	- How accurately can we estimate the effect of each medium on sales?
# 	- How accurately can we predict future sales?
# 	- Is the relationship linear?
# 	We want to find a function that given input budgets for TV, radio and newspaper predicts the output sales
# 	and visualize the relationship between the features and the response using scatter plots.
# 	The objective is to use linear regression to understand how advertisement spending impacts sales.
	
# 	### Data Description
# 	TV
# 	Radio
# 	Newspaper
# 	Sales
# """)


#st.subheader("Checkbox")
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data file", type=["csv"])
    st.sidebar.markdown("""
[Example dataset](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")
    
st.sidebar.title("2. Operations on the Dataset")
w1 = st.sidebar.checkbox("show table", False)
linechart=st.sidebar.checkbox("Linechart",False)
plot= st.sidebar.checkbox("show plots", False)
plothist= st.sidebar.checkbox("show hist plots", False)
trainmodel= st.sidebar.checkbox("Train model", False)
dokfold= st.sidebar.checkbox("DO KFold", False)
Forecast= st.sidebar.checkbox("Forecast", False)



#st.write(w1)

st.subheader('1. Dataset')


######---------------- Machine learning Part ------------------------###################
# Model building

def build_model(df):
    st.write("""
# Machine Learning Implementation
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters!
# Note!
1. This is only for testing pouporses and as such, some features may note work correctly until development is complete. 
2. Data used is also just for testing and does not represent the final accurate data to be used in the model.
""")
    
    #df["Numbers"] = [float(str(i).replace(",", "")) for i in df["Numbers"]]
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y
    #df = df.replace(r'^\s*$', np.nan, regex=True)
    

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

    # Data splitting 
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)
    

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)
    

    #Data splitting
    
    
    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test) )

    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

    

#---------------------------------#

# Sidebar - Specify parameter settings
if trainmodel:
 with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

 with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

 with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    

    def plot_raw_data():
     fig = go.Figure()
     fig.add_trace(go.Scatter(x=df['DATE'], y=df['EF'], name='emission_factors'))
     fig.add_trace(go.Scatter(x=df['DATE'], y=df['AvgTemp'], name='average_temp'))
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='stock_high'))
    # fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='stock_low'))
     fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
     

     tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
     with tab1:
      # Use the Streamlit theme.
       # This is the default. So you can also omit the theme argument.
      st.plotly_chart(fig, theme="streamlit", use_container_width=True)
      with tab2:
      # Use the native Plotly theme.
       st.plotly_chart(fig, theme=None, use_container_width=True)

    plot_raw_data()

    df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

    st.map(df)


    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    st.write('CSV file file must have columns of; "DATE", "AvgTemp", "MinTemp", "MaxTemp", "Total", "Local", "Exotic", "EF"')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Diabetes dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)

######---------------- Machine learning Part END ------------------------###################


@st.cache
def read_data():
    return pd.read_csv(uploaded_file)
df.dropna(inplace= True)
# df = df[df['ds'].notna()]
df.reset_index(drop=True, inplace=True)
df=df[["DATE","EF"]]

#st.dataframe(df)

df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
df.tail()


#st.write(df)

#st.write(df)
if w1:
    st.dataframe(df)


if linechart:
	st.subheader("Line chart")
	st.line_chart(
    df,
    x="ds",
    y=["y"],  # <-- You can pass multiple columns!

)
if plothist:
    st.subheader("Distributions of each Parameter")
    options = ("AvgTemp", "MinTemp", "Annual Growth Rate", "MaxTemp", "Total", "Local", "Exotic", "EF")
    sel_cols = st.selectbox("select columns", options,1)
    st.write(sel_cols)
    #f=plt.figure()
    fig = go.Histogram(x=df[sel_cols],nbinsx=50)
    st.plotly_chart([fig])
    

#    plt.hist(df[sel_cols])
#    plt.xlabel(sel_cols)
#    plt.ylabel("sales")
#    plt.title(f"{sel_cols} vs Sales")
    #plt.show()	
#    st.plotly_chart(f)

if plot:
    st.subheader("correlation between Parameters")
    options = ("AvgTemp", "MinTemp", "MaxTemp", "Total", "Local", "Exotic", "EF")
    w7 = st.selectbox("Ad medium", options,1)
    st.write(w7)
    f=plt.figure()
    plt.scatter(df[w7],df["DATE"])
    plt.xlabel(w7)
    plt.ylabel("sales")
    plt.title(f"{w7} vs DATE")
    #plt.show()	
    st.plotly_chart(f)

    

   

# trainmodel= st.checkbox("Train model", False)
if trainmodel:
	st.header("Modeling")
	y=df.ds
	X=df[["y"]].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	lrgr = LinearRegression()
	lrgr.fit(X_train,y_train)
	pred = lrgr.predict(X_test)

	mse = mean_squared_error(y_test,pred)
	rmse = sqrt(mse)

	st.markdown(f"""
	Linear Regression model trained :
		- MSE:{mse}
		- RMSE:{rmse}
	""")
	st.success('Model trained successfully')

if Forecast:
	st.header("Forecast")
n_years = st.slider("Years of prediction:", 1 , 10)
period = n_years * 365
        
df_train = df[['ds','y']]
df_train = df_train.rename(columns={"ds": "ds", "y": "y"})
df.reset_index(drop=True)

m=Prophet(growth='linear')
m.fit(df_train)
future = m.make_future_dataframe(periods=period) # freq='M'
forecast = m.predict(future)

st.subheader('Forecast data')
#st.write(forecast.tail())

fig2 = plot_plotly(m, forecast)
st.plotly_chart(fig2)

st.subheader('Forecast Components')
fig4 = m.plot_components(forecast)
st.write(fig4)


if dokfold:
	st.subheader("KFOLD Random sampling Evalution")
	st.empty()
	my_bar = st.progress(0)

	from sklearn.model_selection import KFold

	X=df.values[:,-1].reshape(-1,1)
	y=df.values[:,-1]
	#st.progress()
	kf=KFold(n_splits=10)
	#X=X.reshape(-1,1)
	mse_list=[]
	rmse_list=[]
	r2_list=[]
	idx=1
	fig=plt.figure()
	i=0
	for train_index, test_index in kf.split(X):
	#	st.progress()
		my_bar.progress(idx*10)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		lrgr = LinearRegression()
		lrgr.fit(X_train,y_train)
		pred = lrgr.predict(X_test)
		
		mse = mean_squared_error(y_test,pred)
		rmse = sqrt(mse)
		r2=r2_score(y_test,pred)
		mse_list.append(mse)
		rmse_list.append(rmse)
		r2_list.append(r2)
		plt.plot(pred,label=f"dataset-{idx}")
		idx+=1
	plt.legend()
	plt.xlabel("Data points")
	plt.ylabel("PRedictions")
	plt.show()
	st.plotly_chart(fig)

	res=pd.DataFrame(columns=["MSE","RMSE","r2_SCORE"])
	res["MSE"]=mse_list
	res["RMSE"]=rmse_list
	res["r2_SCORE"]=r2_list

	st.write(res)
	st.balloons()
#st.subheader("results of KFOLD")

#f=res.plot(kind='box',subplots=True)
#st.plotly_chart([f])



