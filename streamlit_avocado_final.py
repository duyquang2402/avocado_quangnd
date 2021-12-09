import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime

import pickle

from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

from lazypredict.Supervised import LazyClassifier

data = pd.read_csv('avocado.csv', index_col=0)

# GUI
st.title('Data Science Project')
st.write('## Hass Avocado Price Prediction')

# Upload file
#uploaded_file = st.file_uploader('Choose a file',type=['csv'])
#if uploaded_file is not None:
#    data = pd.read_csv(uploaded_file)
#    data.to_csv('avocado_new.csv',index=False)

df = data.copy()

#Feature Engineering & Preprocessing
def to_season(month):
    if month >= 3 and month <= 5:
        return 0
    elif month >= 6 and month <= 8:
        return 1
    elif month >= 9 and month <= 11:
        return 2
    else:
        return 3

def data_preprocessing(df_new):

    #chuyển dữ liệu biến Date về kiểu datetime
    df_new['Date'] = pd.to_datetime(df_new['Date'])
    #tạo ra cột Month
    df_new['Month'] = pd.DatetimeIndex(df_new['Date']).month
    df_new['year'] = pd.DatetimeIndex(df_new['Date']).year


    #tạo ra cột Season
    df_new['Season'] = df_new['Month'].apply(lambda x: to_season(x))
    #label encoder cho biến type
    le = LabelEncoder()
    df_new['type_new'] = le.fit_transform(df_new['type'])
    #onehot encoder cho biến region
    df_ohe = pd.get_dummies(data=df_new, columns=['region'])
    return df_ohe
#lựa chọn thuộc tính, bỏ đi các thuộc tính dư thừa
#chưa bỏ Total Bags, sẽ xem xét sau khi tìm được mô hình phù hợp
df_ohe = data_preprocessing(df)
y = df_ohe['AveragePrice']
X = df_ohe.drop(['Date','AveragePrice','4046','4225','4770','Small Bags','Large Bags','XLarge Bags','type'], axis=1)

#Modeling & Evaluation / Analyze & Report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Load the Model back from file
Pkl_Filename = "pipe_BAG_Avocado_yc1_Model.pkl"  
with open(Pkl_Filename, 'rb') as file:  
    Avocado_yc1_Model = pickle.load(file)

y_pred_BAG = Avocado_yc1_Model.predict(X_test)
mae_BAG = mean_absolute_error(y_test, y_pred_BAG)
r2_BAG = r2_score(y_test, y_pred_BAG)
train_BAG = Avocado_yc1_Model.score(X_train,y_train)
test_BAG = Avocado_yc1_Model.score(X_test,y_test)
#trục quan hóa dữ liệu
final_yc1 = pd.DataFrame({'Actual': y_test.values,\
                   'Prediction': pd.DataFrame(y_pred_BAG)[0].values})

#Seasonal Prediction
def hass_future_forecast(data, state, hass_type, years):
    #Select region == total us
    df = data[(data['region'] == state) & (data['type'] == hass_type)]
    #tạo thêm biến doanh thu trung bình
    df['AverageRevenue'] = df['AveragePrice'] * df['Total Volume']
    #Select needed columns
    df = df[['Date', 'AverageRevenue']]
    df.columns = ['ds', 'y']
    df.loc[:,'ds'] = pd.to_datetime(df['ds'])
    #Group by date, who analysis cover all two types of hass avocado
    #df = df.groupby(by='ds', ).sum()
    #Scaling data for boost performance and readability
    df['y'] = df['y']/10e6 #Scale to million
    #Reset index for fbprophet
    df = df.reset_index()
    #Fbprophet
    m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    m.fit(df)
    #Create future stream
    future = m.make_future_dataframe(periods=12*years, freq='MS', include_history=True)
    #Forecast the unknown
    forecast = m.predict(future)

    return m, forecast
#_______________________
menu = ['Bussiness Objective','Price Regression - Build Project','Price Regression - Show Prediction', 'FbProphet Model - Cali Organic', 'FbProphet - Show Prediction']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Bussiness Objective':
    st.subheader('Business Objective/Problem')
    st.write('''
    ##### Bơ “Hass”, một công ty có trụ sở tại Mexico, chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ. Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại Bơ đang có cho việc trồng bơ ở các vùng khác. 
    ''')
    st.write('''
    ##### => Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ => xem xét việc mở rộng sản xuất, kinh doanh.
    ''')
    st.image('Hass_avocado_2.jpg')
    st.write('''
    #### Project thực hiện các công việc sau:
    ''')
    st.image('Slide01.png')
elif choice == "Price Regression - Build Project":
    st.subheader("Dữ Liệu Input Để Build Model")
    st.dataframe(data.head(5))
    st.dataframe(data.tail(5))
    st.subheader("Data Cleaning & Preprocessing")
    st.dataframe(df_ohe.head(5))
    st.dataframe(df_ohe.tail(5))
    st.subheader("Build Model - BaggingRegressor")
    st.table(final_yc1.head(10))
    st.table(final_yc1.tail(10))
    st.write('Mean AveragePrice of Dataset:', df['AveragePrice'].mean())
    #st.write('Mean AveragePrice of Testset:', y_test['AveragePrice'].mean())
    st.write('R2 - BaggingRegressor:',r2_BAG)
    st.write('MAE - BaggingRegressor:',mae_BAG)
    st.write('BaggingRegressor Score: Train: ',train_BAG, ' vs Test: ', test_BAG)

    st.write("##### Actual Price Vs Predicted Price")

    fig1, ax1 = plt.subplots()
    ax1 = sns.kdeplot(final_yc1['Actual'], color='r', label='Actual Test Value')
    sns.kdeplot(final_yc1['Prediction'], color='b', label='Predicted Test Value', ax=ax1)
    plt.legend()
    st.pyplot(fig1)

    st.markdown("## Kết luận:")
    st.write("#### Mô hình dự đoán giá trị AveragePrice của Avocado trên toàn nước Mỹ cho kết quả tốt, đạt độ chính xác 90% với độ lệch trung bình khoảng 7% so với thực tế")

elif choice == "Price Regression - Show Prediction":
    test_data_yc1 = pd.DataFrame() #chứa dataframe input
    input_type = "" # để kiểm tra loại input

    st.write("### Dự đoán giá bằng cách upload dữ liệu theo định dạng csv")
    uploaded_file2 = st.file_uploader('Upload new csv file',type=['csv'])
    if uploaded_file2 is not None:
        test_data_yc1 = pd.read_csv(uploaded_file2, index_col=0)
        test_data_yc1.to_csv('avocado_test.csv',index=False) 
        input_type = "csv"       

    st.write("### Dự đoán giá bằng cách nhập liệu")
    states_list = data['region'].unique().tolist()
    hass_type = data['type'].unique().tolist()
    yc1_date = st.date_input("Thời gian:")
    yc1_totalvolume = st.text_input("Total Volume:")
    yc1_totalbag = st.text_input("Total Bags:")
    yc1_type = st.selectbox("Type:", options=hass_type)
    yc1_state = st.selectbox("Region", options=states_list)
    submit_yc1 = st.button('Submit')
    if submit_yc1:
        # List1 
        lst = [[yc1_date, 0, yc1_totalvolume,0,0,0,yc1_totalbag,0,0,0,yc1_type,0,yc1_state]]
        cols = ['Date','AveragePrice','Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags','type','year','region']    
        test_data_yc1 = pd.DataFrame(lst, columns =cols)
        input_type = "form"
    
    if not test_data_yc1.empty:
        st.write("#### Dữ liệu input - head(5)")
        st.table(test_data_yc1.head(5))
        #thêm vào dữ liệu gốc để preprocessing
        df1 = data.copy()
        df1 = df1.append(test_data_yc1)
        df1.reset_index(inplace=True, drop=True)
        df1_ohe_yc1 = data_preprocessing(df1)
        #chỉ lấy dữ liệu mới sau khi đã thực hiện data_preprocessing
        df_test_yc1 = df1_ohe_yc1.iloc[-len(test_data_yc1):]

        y_new_yc1 = df_test_yc1['AveragePrice']
        X_new_yc1 = df_test_yc1.drop(['Date','AveragePrice','4046','4225','4770','Small Bags','Large Bags','XLarge Bags','type'], axis=1)
        y_predict_yc1= Avocado_yc1_Model.predict(X_new_yc1)

        if input_type == "csv":

            #kết quả đự đoán giá
            final_predict_yc1 = pd.DataFrame({'Actual': y_new_yc1.values,\
                            'Prediction': pd.DataFrame(y_predict_yc1)[0].values})
            
            st.subheader("Kết Quả Dự Đoán Giá")
            st.dataframe(final_predict_yc1.head(5))
            st.dataframe(final_predict_yc1.tail(5))

            st.write("##### Actual Price Vs Predicted Price")

            fig2, ax1 = plt.subplots()
            ax1 = sns.kdeplot(final_predict_yc1['Actual'], color='r', label='Actual Test Value')
            sns.kdeplot(final_predict_yc1['Prediction'], color='b', label='Predicted Test Value', ax=ax1)
            plt.legend()
            st.pyplot(fig2)
        elif input_type == "form":
            st.subheader("Kết Quả:")
            st.write("#### Dự Đoán AveragePrice Theo Dữ Liệu Input: ",y_predict_yc1[0])


elif choice == "FbProphet Model - Cali Organic":
    df2 = data.copy()
    df2 = df2[['Date','AveragePrice','region','type']]
    df_cali = df2[(df2['region'] == 'California') & (df2['type'] == 'organic')]
    df_cali = df_cali.drop(['region','type'], axis=1)
    df_cali.columns = ['ds','y']
    df_cali.sort_values('ds', inplace=True)
    df_cali.reset_index(inplace=True, drop=True)
    st.write("#### Dữ Liệu Organic Avocado tại California")
    st.table(df_cali.head(5))
    st.table(df_cali.tail(5))

    st.subheader("Tổng dữ liệu là 169 tuần => Xây dựng model với train 135 tuần và test 34 tuần")
    #chia dữ liệu 80-20
    #tổng dữ liệu là 169 tuần => train 135 tuần test 34 tuần
    train = df_cali.drop(df_cali.index[-34:])
    test = df_cali.drop(df_cali.index[0:-34])
    model_fb = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model_fb.fit(train)

    #34 tuần test + 34 tuần dự đoán
    start_week = datetime.datetime.strptime("2017-08-06","%Y-%m-%d")
    end_week = start_week + datetime.timedelta(weeks=68)
    weeks = pd.date_range(start_week,end_week, freq='W').strftime("%Y-%m-%d").tolist()
    future = pd.DataFrame(weeks)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
    forecast = model_fb.predict(future)

    st.write("#### Kết Quả Dự Đoán")
    st.table(forecast[['ds','yhat']].head(5))
    st.table(forecast[['ds','yhat']].tail(5))

    
    y_test = test['y'].values
    y_pred = forecast['yhat'].values[:34]
    mae_p = mean_absolute_error(y_test, y_pred)
    rmse_p = sqrt(mean_squared_error(y_test, y_pred))
    y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']),columns=['Actual'])
    y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']),columns=['Prediction'])

    # Visulaize the result
    fig3, ax = plt.subplots()
    ax.plot(y_test_value, label='Real AvgPrice')
    ax.plot(y_pred_value, label='Prediction AvgPrice')
    plt.xticks(rotation='vertical')
    ax.legend()
    st.pyplot(fig3)

    st.write("#### 34 tuần test + 34 tuần dự đoán")
    fig4 = model_fb.plot(forecast)   
    a = add_changepoints_to_plot(fig4.gca(), model_fb, forecast)
    st.pyplot(fig4)

    st.write("##### Mean Price of Cali: %.3f" % df_cali.y.mean())
    st.write("##### Mean Price of test: %.3f" % test.y.mean())
    st.write('##### MAE of FbProphet Model: %.3f' % mae_p)
    st.write('##### RMSE of FbProphet Model: %.3f' % rmse_p)

    st.write("## Kết luận:")
    st.write("### Mô hình cho MAE chỉ khoảng 10% so với Mean của dữ liệu, RMSE cũng cho kết quả nhỏ")
    st.write("### => FbProphet phù hợp để dự đoán giá theo seasonal cho bộ dữ liệu này")

elif choice == 'FbProphet - Show Prediction':
    st.subheader("Dự Báo Xu Hướng Thị Trường Hass Avocado")
    st.write("#### Kết Quả Dự Báo Dựa Trên Doanh Thu")
    df3 = data.copy()
    states_list = df3['region'].unique().tolist()
    hass_type = df['type'].unique().tolist()
    region_select = st.selectbox('Region', options=states_list)
    type_select = st.radio('Type', options=hass_type)
    next_years = st.slider('Year', 1,5,1)
    submit = st.button('Submit')

    if submit:
        model_yc4, forecast_yc4 = hass_future_forecast(data=df3, state = region_select, hass_type=type_select, years=next_years)
        #Take a look at the graph
        fig5 = model_yc4.plot(forecast_yc4)
        plot_title = region_select + " | " + type_select + " Avocado Next " + str(next_years) + " Years Prediction"
        plt.title(plot_title, fontsize=20,color="teal")
        a = add_changepoints_to_plot(fig5.gca(), model_yc4, forecast_yc4)
        st.pyplot(fig5)
