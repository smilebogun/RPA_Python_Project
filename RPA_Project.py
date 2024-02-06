#
# 주식가격 시계열 예측 앱.

# 필요한 라이브러리를 불러온다.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression              # 선형회귀 모형.
import warnings
warnings.filterwarnings("ignore")                              # 성가신 warning을 꺼준다.

# # 나눔글꼴 경로 설정
font_path ='c:\\Users\\302-05\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumGothic.ttf'
# # 폰트 이름 가져오기
font_name = fm.FontProperties(fname=font_path).get_name()
# 폰트 설정
plt.rc('font', family=font_name)
# print(matplotlib.get_cachedir())

list =[]
df = pd.read_excel('0200.환율CodeTable.xlsm', sheet_name=['일일환율', '최근20일환율', '선형회귀'])
#df = df.transpose()
df1 = df['일일환율']
df2 = df['최근20일환율']
del df2['통화명']   # 통화명 인덱스 삭제
#df2 = df2.transpose()
for a in df2.columns:
    list.append(pd.to_datetime(a-2, unit='D', origin=pd.to_datetime('1900/01/01')))

df2.columns = list
df2 = df2.transpose()
df2.columns = df1['통화명']
df2= df2.sort_index()
data = df2[:5].copy()

def addARPrediction(columns,pred_ndays, marker,line_style):
    global df2
    df2['m1'] = df2[columns].shift(1)                    # t-1 값.
    df2['m2'] = df2[columns].shift(2)                    # t-2 값.
    df2['m3'] = df2[columns].shift(3)                    # t-3 값.
    df2['m4'] = df2[columns].shift(4)                    # t-4 값.
    df2['m5'] = df2[columns].shift(5)                    # t-5 값.
    df2 = df2.iloc[5:]

    model = LinearRegression()
    model.fit(df2[['m1','m2','m3','m4','m5']],df2[columns])

    rdf = df2[columns][-5:]
    
    n = len(df2)

    for step in range(pred_ndays):             
        # 최신 5개값으로 데이터프레임을 만든다.
            past = pd.DataFrame(data={ f'm{i}': [rdf.iloc[-i]] for i in range(1,6)} ) 

            # 예측 결과는 원소가 1개인 데이터프레임.
            predicted = model.predict(past)[0]  
                                        
            # 예측값과 합치기
            rdf = pd.concat( [rdf, pd.Series({n + step:predicted}) ])

    df5 = pd.DataFrame(rdf.reset_index(drop=True))

    df5 = df5[-pred_ndays:].transpose()
    print(df5)

    date = []
    for d in range(-1,pred_ndays-1):
        date.append(datetime.now()+timedelta(days=d))
    df5.columns = date
    df5 = df5.transpose()
    data2 = []
    data2.append(columns)
    df5.columns = data2
    print(df5)

    plt.plot(df5.index,df5,marker=marker,linestyle=line_style, color="red")

#print(df2)

# 세션 상태를 초기화 한다.
if 'code_index' not in st.session_state:
    st.session_state['code_index'] = '미국 달러 (USD)'

if 'ndays' not in st.session_state:
    st.session_state['ndays'] = 30

if 'chart_color' not in st.session_state:
    st.session_state['chart_color'] = '#1f77b4'

if 'line_style' not in st.session_state:
    st.session_state['line_style'] = 'solid'

if 'chart_marker' not in st.session_state:
    st.session_state['chart_marker'] = 'none'

if 'volume2' not in st.session_state:
    st.session_state['volume2'] = True

if 'volume' not in st.session_state:
    st.session_state['volume'] = True

if 'pred_ndays' not in st.session_state:
    st.session_state['pred_ndays'] = 5

data = []
for a in df2.columns:
    data.append(a)

# 사이드바에서 폼을 통해서 차트 인자를 설정한다.
with st.sidebar.form(key="chartsetting", clear_on_submit=True):
    st.header('차트 설정')
    ''
    ''
    choices = data
    choice = st.selectbox(label='통화명:', options = choices, index=data.index(st.session_state['code_index']))
    # code_index = choices(choice)
    # code = choice.split()[0]                        # 실제 code 부분만 떼어 가져온다.
    ''
    ''
    ndays = st.slider(
        label='데이터 기간 (days):', 
        min_value= 10,
        max_value= 100, 
        value=st.session_state['ndays'],
        step = 10)
    ''
    ''
    chart_colors = ['#1f77b4','red', 'blue' ,'green' ,'black','white' ,'pink']
    chart_color = st.selectbox(label='차트 색상:',options=chart_colors,index=chart_colors.index(st.session_state['chart_color']))
    line_styles = ['solid','dotted','dashed','dashdot']
    line_style = st.selectbox(label='라인 스타일:',options=line_styles,index=line_styles.index(st.session_state['line_style']))
    chart_markers = ['none','o', '*' ,'+' ,'x','D' ,'.','_']
    chart_marker = st.selectbox(label='마커:',options=chart_markers,index=chart_markers.index(st.session_state['chart_marker']))
    ''
    ''
    pred_ndays = st.slider(
        label='예측 기간 (days):', 
        min_value= 1,
        max_value= 10, 
        value=st.session_state['pred_ndays'],
        step = 1)
    
    '---'
    volume2 = st.checkbox('예측값 상세보기', value=st.session_state['volume2'])
    '---'
    volume = st.checkbox('현재 매매기준 환율', value=st.session_state['volume'])
    '---'

    #submitted = st.form_submit_button("Submit")
    
    if st.form_submit_button(label="OK"):
        st.session_state['ndays'] = ndays
        st.session_state['code_index'] = choice
        st.session_state['chart_color'] = chart_color
        st.session_state['line_style'] = line_style
        st.session_state['chart_marker'] = chart_marker
        st.session_state['volume2'] = volume2
        st.session_state['volume'] = volume
        st.session_state['pred_ndays'] = pred_ndays
        st.experimental_rerun()

# chart_title = choices[st.session_state['code_index'] ]
# st.markdown(f'<h3 style="text-align: center; color: red;">{chart_title}</h3>', unsafe_allow_html=True)

# 불러올 날짜 설정
df2 = df2.truncate(before=df2.index[99-ndays])
## 1번 차트
fig = plt.figure(figsize=(10,5))
plt.plot(df2.index,df2[st.session_state['code_index']], linestyle=line_style, color=chart_color, marker=chart_marker)
addARPrediction(columns=st.session_state['code_index'],pred_ndays=st.session_state['pred_ndays'],marker=st.session_state['chart_marker'],line_style=st.session_state['line_style'])
plt.ylabel('환율(원)')
plt.xticks()
plt.title(st.session_state['code_index'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
st.pyplot(fig)    # streamlit에 불러오기

# 2번차트 예측
#################################################################################################
if volume2:
    df3= pd.read_excel('0200.환율CodeTable.xlsm', sheet_name=['일일환율', '최근20일환율', '선형회귀'], index_col=0)
    df3 = df3['최근20일환율']
    df3 = df3.transpose()
    df3.index = pd.to_datetime(df3.index-2, unit='D', origin=pd.to_datetime('1900/01/01'))
    df3 = df3.sort_index()
    print(df3.index)
    data = df3.copy()

    # df3['m1'] = df3['2024-01-30']                    # t-1 값.
    # df3['m2'] = df3['2024-01-29']                    # t-2 값.
    # df3['m3'] = df3['2024-01-28']                    # t-3 값.
    # df3['m4'] = df3['2024-01-27']                    # t-4 값.
    # df3['m5'] = df3['2024-01-26']                    # t-5 값.  

    df3['m1'] = df3[choice].shift(1)                    # t-1 값.
    df3['m2'] = df3[choice].shift(2)                    # t-2 값.
    df3['m3'] = df3[choice].shift(3)                    # t-3 값.
    df3['m4'] = df3[choice].shift(4)                    # t-4 값.
    df3['m5'] = df3[choice].shift(5)                    # t-5 값.  
    df3 = df3.iloc[5:]

    model = LinearRegression()
    model.fit(df3[['m1','m2','m3','m4','m5']], df3[choice])

    # 최신 5일치 데이터
    rdf = df3[choice][-5:]
    n = len(df3)

    # for step in range(5):                     # 미래 예측.
    #   past = pd.DataFrame(data={ f'm{i}': [rdf.iloc[-i]] for i in range(1,6)} ) # 최신 5개값으로 데이터프레임을 만든다.
    #   predicted = model.predict(past)[0]                                        # 예측 결과는 원소가 1개인 데이터프레임.
    #   rdf = pd.concat( [rdf, pd.Series({n + step:predicted}) ])

    # pred_ndays = 5

    # 미래 예측.
    for step in range(pred_ndays):             
        # 최신 5개값으로 데이터프레임을 만든다.
        past = pd.DataFrame(data={ f'm{i}': [rdf.iloc[-i]] for i in range(1,6)} ) 
        # 예측 결과는 원소가 1개인 데이터프레임.
        predicted = model.predict(past)[0]                                       
        # 예측값과 합치기
        rdf = pd.concat( [rdf, pd.Series({n + step:predicted}) ])



    df4 = data['미국 달러 (USD)'].reset_index(drop=True)
    df4.plot(legend=True, label="past")
    df5 = rdf.reset_index(drop=True)
    df5.plot(legend=True, label="Prediction", color="red")
    df5 = df5[-5:]

    # x= np.arange(101,105,1)
    # y= np.sin(x)
    # plt.plot(x,y)
    # plt.show()

    # fig3, ax = plt.plot(
    #     data,
    #     volume=st.session_state['volume'],
    #     style=mpf_style,
    #     figsize=(10,7),
    #     fontscale=1.1,
    #     returnfig=True                  # Figure 객체 반환.
    # )

    # ax.plot(rdf, color = 'red', marker='o', linestyle ='--', linewidth=1.5, label = 'AR(5)')
    # ax.legend(loc='best')

    fig3 = plt.figure(figsize=(10,5))
    plt.plot(df5.index, df5, color="red")
    #plt.plot(rdf.index, rdf)
    #plt.plot(df4.index, df4, color="red")
    plt.ylabel('환율(원)')
    plt.xticks()
    plt.title(st.session_state['code_index'])
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    st.pyplot(fig3)
#################################################################################################

# 3번 차트
if volume:
    fig2 = plt.figure(figsize=(10,5))
    bar = plt.bar(x=df1['통화명'], height=df1['환율(원)'])
    plt.ylabel('환율(원)')
    plt.ylim(df1['환율(원)'].min())
    plt.xticks(rotation=90)
    # 바 그래프 위에 데이터 값 표시
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 12)
    st.pyplot(fig2)   # streamlit에 불러오기