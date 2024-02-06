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

# # # 나눔글꼴 경로 설정
# font_path = 'NanumGothic.ttf'
# font_path ='c:\\Users\\302-05\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NanumGothic.ttf'
# # 폰트 이름 가져오기
# font_name = fm.FontProperties(fname=font_path).get_name()
font_name = fm.FontProperties('NanumGothic.ttf')
# 폰트 설정
plt.rc('font', family=font_name)
# print(matplotlib.get_cachedir())
plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['font.family'] = 'Malgun Gothic'
# font_list = [font.name for font in fm.fontManager.ttflist]
# print(font_list)

df = pd.read_excel('0200.환율CodeTable.xlsm', sheet_name=['일일환율', '최근20일환율'], index_col=0)
df1 = df['일일환율']
# df1 = df1.transpose()
print(df1)
print(df1.index)
#print(df1.reset_index(drop=True))

df2 = df['최근20일환율']
# del df2['통화명']   # 통화명 인덱스 삭제
df2 = df2.transpose()
print("======================")
print(df2)

df2.index = pd.to_datetime(df2.index-2, unit='D', origin=pd.to_datetime('1900/01/01'))
print("======================")
print(df2.index)

df2 = df2.sort_index()      ## 전체DB : df2 // 오름차순 정렬 ==> 사이드바 셀렉하기 위해 
print("##################")
print(df2.head())


df5 = df2                   ## 한종목 DB 전체기간 : 파일에서 한 종목만 df5로 꺼내오기 100개 => 차트에 뿌릴 변수
final_index = df5.index     ## 날짜 인덱스 넣기 df2로 해도 상관없음 100개 ==> df6.index 로 사용
data = df5.copy()           ## 한종목 DB 전체기간 : df5 한종목을 data로 백업 ==> df6 으로 출력
print('------------------- data -----------')
print(data.head())

# # Streamlit + SideBar
#################################################################################################
# 세션 상태를 초기화 한다.
if 'code_index' not in st.session_state:
    st.session_state['code_index'] = '미국 달러 (USD)'

if 'ndays' not in st.session_state:
    st.session_state['ndays'] = 100

if 'chart_color' not in st.session_state:
    st.session_state['chart_color'] = '#1f77b4'

if 'line_style' not in st.session_state:
    st.session_state['line_style'] = 'solid'

if 'chart_marker' not in st.session_state:
    st.session_state['chart_marker'] = 'none'

if 'volume' not in st.session_state:
    st.session_state['volume'] = True

if 'volume2' not in st.session_state:
    st.session_state['volume2'] = True

if 'volume3' not in st.session_state:
    st.session_state['volume3'] = True

if 'pred_ndays' not in st.session_state:
    st.session_state['pred_ndays'] = 5

# choice 설정
data2 = []
for a in df2.columns:
    data2.append(a)

# 사이드바에서 폼을 통해서 차트 인자를 설정한다.
with st.sidebar.form(key="chartsetting", clear_on_submit=True):
    st.header('차트 설정')
    ''
    st.write("1-1. 통화명 & 데이터기간")
    choices = data2
    choice = st.selectbox(label='통화명:', options = choices, index=data2.index(st.session_state['code_index']))
    # code_index = choices(choice)
    # code = choice.split()[0]    # 실제 code 부분만 떼어 가져온다.
    ndays = st.slider(
        label='데이터 기간 (days):', 
        min_value= 10,
        max_value= 100, 
        value=st.session_state['ndays'],
        step = 10)
    ''
    st.write("1-2. 차트 스타일")
    chart_colors = ['#1f77b4','red', 'blue' ,'green' ,'black', 'pink']
    chart_color = st.selectbox(label='차트 색상:',options=chart_colors,index=chart_colors.index(st.session_state['chart_color']))
    line_styles = ['solid','dotted','dashed','dashdot']
    line_style = st.selectbox(label='라인 스타일:',options=line_styles,index=line_styles.index(st.session_state['line_style']))
    chart_markers = ['none','o', '*' ,'+' ,'x','D' ,'.','_']
    chart_marker = st.selectbox(label='마커: (1, 2차트 적용)',options=chart_markers,index=chart_markers.index(st.session_state['chart_marker']))
    ''
    volume3 = st.checkbox('Chart_Grid (1, 2차트 적용)', value=st.session_state['volume3'])
    ''
    st.write("1-3. 예측 기간")
    pred_ndays = st.slider(
        label='예측 기간 (days):', 
        min_value= 0,
        max_value= 10, 
        value=st.session_state['pred_ndays'],
        step = 5)
    '---'
    volume2 = st.checkbox('2_예측값 상세보기', value=st.session_state['volume2'])
    st.write("#. 예측기간이 0일때 비활성화")
    '---'
    volume = st.checkbox('3_현재 매매기준 환율', value=st.session_state['volume'])
    #submitted = st.form_submit_button("Submit")
    '---'
    if st.form_submit_button(label="OK"):
        st.session_state['ndays'] = ndays
        st.session_state['code_index'] = choice
        st.session_state['chart_color'] = chart_color
        st.session_state['line_style'] = line_style
        st.session_state['chart_marker'] = chart_marker
        st.session_state['volume'] = volume
        st.session_state['volume2'] = volume2
        st.session_state['volume3'] = volume3
        st.session_state['pred_ndays'] = pred_ndays
        if pred_ndays == 0:
            volume2 = False
            st.session_state['volume2'] = volume2
        st.experimental_rerun()
#################################################################################################
        
# 예측기간 설정        
date = []
for d in range(0, pred_ndays+1):
    # 실시간 데이터를 사용할 경우 => 
    # 아래 rdf.index = pd.to_datetime(rdf.index-2, unit='D', origin=pd.to_datetime('1900/01/01'))
    # 가 적힌 214번, 305번 주석처리
    #date.append(datetime.now()+timedelta(days=d))  
    date.append(45322 + d)
    
df5['m1'] = df2[st.session_state['code_index']].shift(1)                    # t-1 값.
df5['m2'] = df2[st.session_state['code_index']].shift(2)                    # t-2 값.
df5['m3'] = df2[st.session_state['code_index']].shift(3)                    # t-3 값.
df5['m4'] = df2[st.session_state['code_index']].shift(4)                    # t-4 값.
df5['m5'] = df2[st.session_state['code_index']].shift(5)                    # t-5 값.
df5 = df5.iloc[5:]      ## 쉬프트된 데이터 맨 위 5개를 잘라서 출력(결측치 없애기) 총 95
print(df5)              # df5 총 100개 데이터

# 선형회귀 기반  AR(5)모형 학습.
model = LinearRegression()
model.fit(df5[['m1','m2','m3','m4','m5']], df5[st.session_state['code_index']]) # 100개 데이터 확인

# 최신 5일치 데이터
rdf = df5[st.session_state['code_index']][-5:]   # 아래 5개 데이터 (1/27~1/31)
print("^^^^^^^^^")
print(rdf)

n = len(data)       # data는 한종목 => len 길이 100개
print("data--------")
print(data) # 한종목 100개 데이터

# pred_ndays = 5
# 미래 예측.
for step in range(pred_ndays):             
    # 최신 5개값으로 데이터프레임을 만든다.
    past = pd.DataFrame(data={ f'm{i}': [rdf.iloc[-i]] for i in range(1,6)} ) 
    print("past--------")   
    print(past)             # 과거 1/27~1/31 5개 데이터
    # 예측 결과는 원소가 1개인 데이터프레임.
    predicted = model.predict(past)[0]  # 예측 1개 predicted :  1328.4946862462914
    print("predicted : ", predicted)                                     
    # 예측값과 합치기
    rdf = pd.concat( [rdf, pd.Series({n + step:predicted}) ])
    print("+++++++++++")
    print(rdf)      # len길이 100을 인덱스로 뒤에 예측데이터 합치기

# data는 23년 10월 24일 ~ 24년 1월 ~31일 100개 데이터 
print(data[st.session_state['code_index']].reset_index(drop=True))
df6 = data[st.session_state['code_index']].reset_index(drop=True)
# 인덱스(날짜전부)와 칼럼을(통화명(미국)) 전부삭제하고 인덱스에 임의로 0~99 넣어줌(칼럼은 없음)
# 그리고 data(한종목 100개 데이터)를 df6.index 에 넣는다 
print("123123123123123")
print(final_index)
df6.index = final_index # 인덱스 날짜로 변경
df6.plot(legend=True, label="past")
print("00000000000000000")
print(df6)

# 과거 5일 + 예측 5일 ==> 총 10일 셋팅
rdf = rdf.reset_index(drop=True)
print("1111111111111111111")
print(rdf)
rdf = rdf[4:]
# rdf.reset_index(pd.Index([95, 96, 97, 98, 99, 100, 101, 102, 103, 104]))
# rdf.set_index(pd.Index(['1번', '2번', '3번', '4번']))
# rdf.index = [95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
# rdf.index = ['2024-01-27', '2024-01-28', '2024-01-29', '2024-01-30', '2024-01-31', '2024-02-01', '2024-02-02', '2024-02-03', '2024-02-04', '2024-02-05']
# rdf.index = [45318, 45319, 45320, 45321, 45322, 45323, 45324, 45325, 45326, 45327]
rdf.index = date
print("2222222222222222222")
print(rdf)
#rdf.index = pd.to_datetime(df.index)
rdf.index = pd.to_datetime(rdf.index-2, unit='D', origin=pd.to_datetime('1900/01/01'))
print("3333333333333333333")
print(rdf)

# 과거 5일 + 예측 5일 ==> 총 10일 출력
df7 = rdf
print("@@@@@@@@@@@@@@@@@@@@")
print(df7) # df7는 1월27일~31일 과거데이터 + 예측데이터 100~104 5개
df7.plot(legend=True, label="Prediction", color="red")
plt.show()

# 메인 타이틀
chart_title = "환율변동추이 분석 & 예측"
st.markdown(f'<h3 style="text-align: center; color: red;">{chart_title}</h3>', unsafe_allow_html=True)
''
''
        
# # 불러올 날짜 설정
df6 = df6.truncate(before=df6.index[100-ndays])
        
# ## 1번 차트
fig = plt.figure(figsize = (10, 7.5))
#plt.plot(df6.index,df6[st.session_state['code_index']], linestyle=line_style, color=chart_color, marker=chart_marker)
#addARPrediction(columns=st.session_state['code_index'],pred_ndays=st.session_state['pred_ndays'],marker=st.session_state['chart_marker'],line_style=st.session_state['line_style'])
plt.plot(df6.index, df6, linestyle=line_style, color=chart_color, marker=chart_marker, label="Past")
if pred_ndays != 0:
    plt.plot(df7.index, df7, color="red", label="Prediction", marker=chart_marker)
# addARPrediction(df6.index, df6[st.session_state['code_index']], linestyle=line_style, color=chart_color, marker=chart_marker)
# addARPrediction(pred_ndays=st.session_state['pred_ndays'],marker=st.session_state['chart_marker'],line_style=st.session_state['line_style'])
plt.ylabel('환율(원)', rotation=0, loc='top')
plt.xlabel('기간(년-월-일)', loc='right')
# plt.xticks() # 기간자동으로 들어가서 필요없는 값
plt.title("1. " + choice, size=20)
plt.legend()
if volume3:
    plt.grid(True)
st.pyplot(fig)
# fig = plt.figure(figsize=(10,5))
# plt.plot(df2.index,df2[st.session_state['code_index']], linestyle=line_style, color=chart_color, marker=chart_marker)
# plt.plot(df2.index, df2)
# plt.ylabel('환율(원)')
# plt.xticks()
# plt.title(st.session_state['code_index'])
# # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# st.pyplot(fig)    # streamlit에 불러오기



# 예 측 ## 2번 차트
#################################################################################################
if pred_ndays != 0:
    if volume2:
        df3= pd.read_excel('0200.환율CodeTable.xlsm', sheet_name=['일일환율', '최근20일환율'], index_col=0)
        df3 = df3['최근20일환율']
        df3 = df3.transpose()
        df3.index = pd.to_datetime(df3.index-2, unit='D', origin=pd.to_datetime('1900/01/01'))
        df3 = df3.sort_index()

        final_index = df3.index
        data3 = df3.copy()

        df3['m1'] = df3[st.session_state['code_index']].shift(1)                    # t-1 값.
        df3['m2'] = df3[st.session_state['code_index']].shift(2)                    # t-2 값.
        df3['m3'] = df3[st.session_state['code_index']].shift(3)                    # t-3 값.
        df3['m4'] = df3[st.session_state['code_index']].shift(4)                    # t-4 값.
        df3['m5'] = df3[st.session_state['code_index']].shift(5)                    # t-5 값.  
        df3 = df3.iloc[5:]

        model = LinearRegression()
        model.fit(df3[['m1','m2','m3','m4','m5']], df3[st.session_state['code_index']])

        # 최신 5일치 데이터
        rdf = df3[st.session_state['code_index']][-5:]
        n = len(data3)

        # pred_ndays = 5
        # 미래 예측.
        for step in range(pred_ndays):             
            # 최신 5개값으로 데이터프레임을 만든다.
            past = pd.DataFrame(data={ f'm{i}': [rdf.iloc[-i]] for i in range(1,6)} ) 
            # 예측 결과는 원소가 1개인 데이터프레임.
            predicted = model.predict(past)[0]                                       
            # 예측값과 합치기
            rdf = pd.concat( [rdf, pd.Series({n + step:predicted}) ])

        # df4 = data3['미국 달러 (USD)'].reset_index(drop=True)
        # df4.index = final_index
        # df4.plot(legend=True, label="past")

        rdf = rdf.reset_index(drop=True)
        rdf = rdf[4:]
        # rdf.index = [45318, 45319, 45320, 45321, 45322, 45323, 45324, 45325, 45326, 45327]
        rdf.index = date
        rdf.index = pd.to_datetime(rdf.index-2, unit='D', origin=pd.to_datetime('1900/01/01'))
        rdf.plot(legend=True, label="Prediction", color="red")
        plt.show()

        fig3 = plt.figure(figsize = (12, 6))
        plt.plot(rdf.index, rdf, color="red", label="Prediction", marker=chart_marker)
        plt.ylabel('환율(원)', rotation=0, loc='top')
        plt.xlabel('기간(년-월-일)', loc='right')
        plt.xticks()
        plt.legend()
        plt.title("2. 예측값 상세보기 : " + st.session_state['code_index'], size=20)
        if volume3:
            plt.grid(True)
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        st.pyplot(fig3)
#################################################################################################

'---'    
# ## 3번 차트

# print(df1['통화명'].reset_index(drop=True))
# df1 = df1['통화명'].reset_index(drop=True)
if volume:
    fig2 = plt.figure(figsize=(10,20))
    bar = plt.barh(y= df1.index, width = df1['환율(원)'])
    plt.ylabel('통화명', rotation=0, loc='top')
    plt.xlabel('환율(원)', rotation=0, loc='right')
    plt.xlim(df1['환율(원)'].min())
    plt.xticks(rotation=0, fontsize=10)
    plt.title('3. 현재 매매기준 환율', size=20)

    # 바 그래프 위에 데이터 값 표시
    for rect in bar:
        width = rect.get_width()
        plt.text(width, rect.get_y()+0.2, '%.2f' %width)
        # (x축 위치, y축 위치, text값)

    st.pyplot(fig2)   # streamlit에 불러오기