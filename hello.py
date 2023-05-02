import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import base64




# Title
st.title('SIRGE Cattle Emissions Prediction App')
st.write(""" 
## Introduction
In this implementation, Streamlit is used to create an application that calculates and predicts cattle emissions. The app accepts user input data manually or in CSV format.
The first form that appears when the app is launched allows the user to enter data manually. The user is asked to input the number of bulls, cows, calves (female and male), steers, heifers, and oxen. After entering the data, the app performs a calculation based on the entered values and displays the result.
The second form allows the user to upload a CSV file containing cattle emission data. Once uploaded, the data is displayed in a table, and the app calculates the emissions based on the data in the file. The calculated emissions are displayed in a line chart, and a bar chart showing the emissions by ZARDI is also displayed.
The third section of the app uses linear regression to predict cattle emissions. The user can see the performance of the model on the test set and the R-squared score. A new dataframe with predicted emissions is also displayed, and a graph comparing the actual emissions and the predicted emissions is shown.""")

# Create a form to accept user input
if st.sidebar.button("Enter data Manually"):

    st.markdown(f'Please enter the number of:')
    a = st.number_input('Bulls:', value=0.0)
    b = st.number_input('Cows:', value=0.0)
    c = st.number_input('Calves Female:', value=0.0)
    d = st.number_input('Calves Male:', value=0.0)
    e = st.number_input('Steers:', value=0.0)
    f = st.number_input('Heifers:', value=0.0)
    g = st.number_input('Oxen:', value=0.0)

    # Perform a calculation
    z= (a * 52.9421548392805) + (b * 41.1936984829892) + (c * 14.6420826883893) + (d * 17.7291651070273) + (e * 48.7973035004382) + (f * 39.6739869697752) + (g * 50.1822857909654)

    # Display the output
    st.markdown(f'The total emission is <p style="color:red"> {z}</p>', unsafe_allow_html=True)

# Create a form to accept user csv file upload

    
data = st.sidebar.file_uploader('Upload your data', type='csv')

# If data is uploaded, display it in a table and perform the calculation
if data:
    df = pd.read_csv(data)
    st.write('Input Data')
    st.dataframe(df)
  
    st.write('Calculated Emissions')
    # Apply the calculation to each row of the table
    df['Emissions'] = (df['Bull'] * 2) + (df['Calves_female'] * 3) + (df['Calves_male'] * 5) + (df['Cows'] * 6) + (df['Hiefers'] * 7) + (df['Oxen'] * 8)# Modify this line to perform your desired calculation
    
    

    st.write('Output Data')
    st.dataframe(df)
    st.line_chart(df['Emissions'])

    # Draw a bar chart of the emissions
    chart = alt.Chart(df).mark_bar().encode(
        x='ZARDIUganda',
        y='Emissions'
    ).properties(
        title='Emissions by ZARDI'
    )
    st.altair_chart(chart, use_container_width=True)

    # Create a line chart of sepal length by species
    fig = px.line(df, x='Year', y='Emissions', color='ZARDIUganda',
                title='Emissions by ZARDI', 
                labels={'x': 'Year', 'y': 'Emissions', 'ZARDI': 'ZARDI'})
    fig.update_traces(mode='markers+lines')
    fig.update_layout(hovermode='x')

    # Show the chart in Streamlit
    st.plotly_chart(fig)


####--------- Linear Regression and Prediction Model ----------###########
    st.markdown("# Creating and Testing the Machine Learning Model")
    # Use get_dummies to one-hot encode categorical variables
    df_encoded = pd.get_dummies(df[['Year', 'ZARDIUganda', 'Emissions']])

    # Split the data into training and testing sets
    X = df_encoded.drop("Emissions", axis=1)
    y = df_encoded["Emissions"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add model parameters
    st.write('#### Model Parameters')
    st.write('##### Test size:', 0.2)
    st.write('##### Random state:', 42)

    # Fit a linear regression model
    model = LinearRegression().fit(X_train, y_train)    

    # Evaluate the performance of the model on the test set
    score = model.score(X_test, y_test)
    st.write("##### R-squared score:", score)

    st.markdown("# Predictions ")
    st.markdown("### Updated Dataset with predicted values ")

    # Use the trained model to make predictions
    y_pred = model.predict(X)

    # Add the predicted values to the dataframe
    df['Emissions_Predicted'] = y_pred

    # Display the updated dataframe
    st.write(df)
    
    st.markdown("### Prediction Graphs ")
    
    # Create a new DataFrame with the predicted emissions
    df_pred = df[['Year', 'Emissions']].copy()
    df_pred['Emissions_Predicted'] = model.predict(X)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the actual and predicted emissions
    sns.lineplot(data=df_pred, x='Year', y='Emissions', label='Actual', ax=ax)
    sns.lineplot(data=df_pred, x='Year', y='Emissions_Predicted', label='Predicted', ax=ax)

    # Set plot title and axis labels
    ax.set_title('Actual vs Predicted Emissions (only current data)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions (metric tons)')

    # Show the plot
    plt.legend()
    st.pyplot(fig)


    # Create a new DataFrame with the predicted emissions
    df_pred = df[['Year', 'Emissions']].copy()
    df_pred['Emissions_Predicted'] = model.predict(X)

    # Create a new dataset with the years we want to predict emissions for
    years_pred = pd.DataFrame({'Year': range(2015, 2036)})

    # Use get_dummies to one-hot encode categorical variables
    df_pred_encoded = pd.get_dummies(years_pred.merge(df[['Year', 'ZARDIUganda']], how='left', on='Year'))

    # Use the trained model to make predictions
    y_pred = model.predict(df_pred_encoded)

    # Add the predicted values to the dataframe
    df_pred_encoded['Emissions_Predicted'] = y_pred

    # Concatenate the original and predicted dataframes
    df_pred = pd.concat([df_pred, df_pred_encoded[['Year', 'Emissions_Predicted']]])

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the actual and predicted emissions
    sns.lineplot(data=df_pred, x='Year', y='Emissions', label='Actual trend', ax=ax)
    sns.lineplot(data=df_pred, x='Year', y='Emissions_Predicted', label='Predicted trend', ax=ax)

    # Set plot title and axis labels
    ax.set_title('Actual vs Predicted Emissions (2015-2036)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions (metric tons)')

    # Show the plot
    plt.legend()
    st.pyplot(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred['Year'], y=df_pred['Emissions'], name='Actual'))
    fig.add_trace(go.Scatter(x=df_pred['Year'], y=df_pred['Emissions_Predicted'], name='Predicted'))
    fig.update_layout(title='Actual vs Predicted Emissions for the Next 10 Years', xaxis_title='Year', yaxis_title='Emissions (metric tons)')
    st.plotly_chart(fig)


    # Generate analysis report
    st.markdown("# Analysis Report")

    # Save analysis report
    st.markdown("## Download Analysis Report")
    html = "<h1>Analysis Report</h1>"
    html += df.to_html()
    filename = "analysis_report.html"
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="{filename}">Download HTML report</a>'
    st.markdown(href, unsafe_allow_html=True)


# Define the calculation function
# base_data = pd.read_csv('src\Uganda_total_emissions.csv')

if st.button("Use Sample Dataset"):
    df = pd.read_csv('Uganda_total_emissions.csv')
    st.write('Input Data')
    st.dataframe(df)
  
    st.write('Calculated Emissions')
    # Apply the calculation to each row of the table
    df['Emissions'] = (df['Bull'] * 2) + (df['Calves_female'] * 3) + (df['Calves_male'] * 5) + (df['Cows'] * 6) + (df['Hiefers'] * 7) + (df['Oxen'] * 8)# Modify this line to perform your desired calculation
    
    

    st.write('Output Data')
    st.dataframe(df)
    st.line_chart(df['Emissions'])

    # Draw a bar chart of the emissions
    chart = alt.Chart(df).mark_bar().encode(
        x='ZARDIUganda',
        y='Emissions'
    ).properties(
        title='Emissions by ZARDI'
    )
    st.altair_chart(chart, use_container_width=True)

    # Create a line chart of sepal length by species
    fig = px.line(df, x='Year', y='Emissions', color='ZARDIUganda',
                title='Emissions by ZARDI', 
                labels={'x': 'Year', 'y': 'Emissions', 'ZARDI': 'ZARDI'})
    fig.update_traces(mode='markers+lines')
    fig.update_layout(hovermode='x')

    # Show the chart in Streamlit
    st.plotly_chart(fig)


####--------- Linear Regression and Prediction Model ----------###########
    st.markdown("# Creating and Testing the Machine Learning Model")
    # Use get_dummies to one-hot encode categorical variables
    df_encoded = pd.get_dummies(df[['Year', 'ZARDIUganda', 'Emissions']])

    # Split the data into training and testing sets
    X = df_encoded.drop("Emissions", axis=1)
    y = df_encoded["Emissions"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add model parameters
    st.write('#### Model Parameters')
    st.write('##### Test size:', 0.2)
    st.write('##### Random state:', 42)

    # Fit a linear regression model
    model = LinearRegression().fit(X_train, y_train)    

    # Evaluate the performance of the model on the test set
    score = model.score(X_test, y_test)
    st.write("##### R-squared score:", score)

    st.markdown("# Predictions ")
    st.markdown("### Updated Dataset with predicted values ")

    # Use the trained model to make predictions
    y_pred = model.predict(X)

    # Add the predicted values to the dataframe
    df['Emissions_Predicted'] = y_pred

    # Display the updated dataframe
    st.write(df)
    
    st.markdown("### Prediction Graphs ")
    
    # Create a new DataFrame with the predicted emissions
    df_pred = df[['Year', 'Emissions']].copy()
    df_pred['Emissions_Predicted'] = model.predict(X)

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the actual and predicted emissions
    sns.lineplot(data=df_pred, x='Year', y='Emissions', label='Actual', ax=ax)
    sns.lineplot(data=df_pred, x='Year', y='Emissions_Predicted', label='Predicted', ax=ax)

    # Set plot title and axis labels
    ax.set_title('Actual vs Predicted Emissions (only current data)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions (metric tons)')

    # Show the plot
    plt.legend()
    st.pyplot(fig)


    # Create a new DataFrame with the predicted emissions
    df_pred = df[['Year', 'Emissions']].copy()
    df_pred['Emissions_Predicted'] = model.predict(X)

    # Create a new dataset with the years we want to predict emissions for
    years_pred = pd.DataFrame({'Year': range(2015, 2036)})

    # Use get_dummies to one-hot encode categorical variables
    df_pred_encoded = pd.get_dummies(years_pred.merge(df[['Year', 'ZARDIUganda']], how='left', on='Year'))

    # Use the trained model to make predictions
    y_pred = model.predict(df_pred_encoded)

    # Add the predicted values to the dataframe
    df_pred_encoded['Emissions_Predicted'] = y_pred

    # Concatenate the original and predicted dataframes
    df_pred = pd.concat([df_pred, df_pred_encoded[['Year', 'Emissions_Predicted']]])

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the actual and predicted emissions
    sns.lineplot(data=df_pred, x='Year', y='Emissions', label='Actual trend', ax=ax)
    sns.lineplot(data=df_pred, x='Year', y='Emissions_Predicted', label='Predicted trend', ax=ax)

    # Set plot title and axis labels
    ax.set_title('Actual vs Predicted Emissions (2015-2036)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions (metric tons)')

    # Show the plot
    plt.legend()
    st.pyplot(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_pred['Year'], y=df_pred['Emissions'], name='Actual'))
    fig.add_trace(go.Scatter(x=df_pred['Year'], y=df_pred['Emissions_Predicted'], name='Predicted'))
    fig.update_layout(title='Actual vs Predicted Emissions for the Next 10 Years', xaxis_title='Year', yaxis_title='Emissions (metric tons)')
    st.plotly_chart(fig)


    # Generate analysis report
    st.markdown("# Analysis Report")

    # Save analysis report
    st.markdown("## Download Analysis Report")
    html = "<h1>Analysis Report</h1>"
    html += df.to_html()
    filename = "analysis_report.html"
    b64 = base64.b64encode(html.encode()).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="{filename}">Download HTML report</a>'
    st.markdown(href, unsafe_allow_html=True)