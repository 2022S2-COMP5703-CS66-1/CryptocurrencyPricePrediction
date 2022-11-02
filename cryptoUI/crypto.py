# Import libraries
import streamlit as st
import pandas as pd
from BinanceClient import *
import dateutil.parser
import os
import glob
import plotly.graph_objects as go

import requests


st.set_page_config(initial_sidebar_state='collapsed',
    page_title="BTC Dashboard",
    page_icon='btc.png'
)

# defining key/request url
key = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

# requesting data from url
data = requests.get(key)
data = data.json()

client = BinanceClient(futures=False)
symbol = "BTCUSDT"
interval = "1m"

st.sidebar.write('Select Live Data From API to Visualise')
fromDate = st.sidebar.date_input('From', datetime.strptime('2022-10-31', '%Y-%m-%d'))

liveDataLimit = st.sidebar.number_input('Input the number of data limit', key=1, min_value=0, value=1000)

st.sidebar.success('From date: `%s`\n\nData limit: `%d`' % (fromDate, liveDataLimit))

st.sidebar.write('--------------------------------------')

fromDate = int(datetime.strptime('{}-{}-{}'.format(fromDate.year, fromDate.month, fromDate.day), '%Y-%m-%d').timestamp() * 1000)
toDate = int(datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d').timestamp() * 1000)



data1 = GetHistoricalData(client, symbol, fromDate, toDate, liveDataLimit)
df111 = GetDataFrame(data1)

print(df111.head)

# use glob to get all the csv files
# in the folder
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
file_names = [path[path.rfind('/')+1:]for path in csv_files]

def load_data(file_name, nrows=1000):
    d = pd.read_csv(file_name, nrows=nrows)
    return d


st.title("Bitcoin Visualisation Dashboard")

curr_price = round(float(data['price']), 2)
st.metric(label="Bitcoin", value="{} USD".format(curr_price), delta=round(curr_price - df111['Close'][df111.index[-1]], 2))

st.write('*expand sitebar to the left for more info*')

file_name = 'minutesprice.csv'


st.sidebar.write('Select Historical Data on File to Visualise')
hisDataLimit = st.sidebar.number_input('Input the number of data limit', key=2, min_value=0, value=1000)

df = load_data(file_name, hisDataLimit)
df1 = df[['DateTime', 'close']]

start = dateutil.parser.parse(df['DateTime'].iloc[-1])
end = dateutil.parser.parse(df['DateTime'].iloc[0])

start_date = st.sidebar.date_input('Start date', start)
end_date = st.sidebar.date_input('End date', end)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`\n\n Data limit: `%d`' % (start_date, end_date, hisDataLimit))
else:
    st.sidebar.error('Error: End date must fall after start date.')

# Line Chart
df1 = df1[(start_date.strftime('%Y-%m-%d ') <= df['DateTime']) & (df['DateTime'] <= end_date.strftime('%Y-%m-%d'))]
df1 = df1.rename(columns={'DateTime': 'index'}).set_index('index')
# st.line_chart(df1)


df2 = df[['DateTime', 'sentiment_score']]
df2 = df2[(start_date.strftime('%Y-%m-%d ') <= df['DateTime']) & (df['DateTime'] <= end_date.strftime('%Y-%m-%d'))]
df2 = df2.rename(columns={'DateTime': 'index'}).set_index('index')
# st.bar_chart(df2)

fig2 = go.Figure()
# plot data
fig2.add_trace(
 go.Bar(x=df2.index, y=df2['sentiment_score'], marker = {'color':'firebrick'})
)


fig2.update_layout(
    autosize=True,
    height=600,
    title="Historical Sentiment Score Visualisation",
    xaxis_title="Days",
    yaxis_title="Score",
    template='plotly_white'
)


df3 = df[['DateTime', 'sentiment_score']]
df3 = df3[(start_date.strftime('%Y-%m-%d ') <= df['DateTime']) & (df['DateTime'] <= end_date.strftime('%Y-%m-%d'))]
df3 = df3.rename(columns={'DateTime': 'index'}).set_index('index')

fig = go.Figure()
fig.add_trace(go.Scatter(mode='lines', x=df1.index, y=df1['close'], line_color='blue', name='Close'))
# fig2.add_trace(go.Scatter(mode='lines', x=df3.index, y=df3['sentiment_score'], line_color='orange', name='Sentiment'))
# fig.add_trace(go.Scatter(mode='lines', x=valid.index, y=valid['Predictions'], line_color='green',name='Predictions'))

fig3 = go.Figure()
fig3.update_layout(
    autosize=True,
    height=600,
    title="Bitcoin Real Time Data Visualisation",
    xaxis_title="Days",
    yaxis_title="Close Price USD ($)",
    template='plotly_white'
)
fig3.add_trace(go.Scatter(mode='lines', x=df111.index, y=df111['Close'], line_color='orange', name='Close'))

# Add range slider
fig3.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="minute",
                     step="minute",
                     stepmode="backward"),
                dict(count=3,
                     label="hour",
                     step="hour",
                     stepmode="backward"),
                dict(count=6,
                     label="day",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="month",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.update_layout(
    autosize=True,
    height=600,
    title="Bitcoin Historical Data Visualisation",
    xaxis_title="Days",
    yaxis_title="Close Price USD ($)",
    template='plotly_white'
)


# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="minute",
                     step="minute",
                     stepmode="backward"),
                dict(count=3,
                     label="hour",
                     step="hour",
                     stepmode="backward"),
                dict(count=6,
                     label="day",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="month",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)


st.write(fig3)

st.write(df111.describe())

st.write('--------------------------------------')

st.write(fig)

st.write(fig2)
st.write(df1.describe())


