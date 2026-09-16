import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="서울 100년 기온 변화 관측",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #E11D48;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_seoul_weather_data():
    """
    GitHub 저장소에서 서울 기온 데이터를 불러와 정제하는 함수입니다.
    """
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    
    # 한국어 CSV 인코딩 호환 처리
    encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-8-sig']
    df = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(url, encoding=enc)
            break
        except Exception:
            continue
            
    if df is None:
        st.error("데이터를 불러오는 중 오류가 발생했습니다.")
        return None
        
    # 공백 제거 및 열 이름 표준화
    df.columns = df.columns.str.strip()
    
    column_mapping = {}
    for col in df.columns:
        if "날짜" in col:
            column_mapping[col] = "날짜"
        elif "평균" in col:
            column_mapping[col] = "평균기온"
        elif "최저" in col:
            column_mapping[col] = "최저기온"
        elif "최고" in col:
            column_mapping[col] = "최고기온"
        elif "지점" in col:
            column_mapping[col] = "지점"
            
    df = df.rename(columns=column_mapping)
    
    # 날짜 파싱 및 결측치 제거
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜', '평균기온'])
    
    # 연도 및 월 추출
    df['연도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    
    return df

df_raw = load_seoul_weather_data()

if df_raw is None or df_raw.empty:
    st.warning("데이터가 비어있거나 불러올 수 없습니다.")
    st.stop()

# 연도별 집계 데이터 생성
yearly_df = df_raw.groupby('연도').agg(
    평균기온=('평균기온', 'mean'),
    최저기온평균=('최저기온', 'mean'),
    최고기온평균=('최고기온', 'mean'),
    최고기온극값=('최고기온', 'max'),
    최저기온극값=('최저기온', 'min'),
    관측일수=('평균기온', 'count')
).reset_index()

# 관측 일수가 너무 적은 해(예: 전쟁 기간 등 데이터 누락 연도) 예외 처리
yearly_df = yearly_df[yearly_df['관측일수'] > 300].copy()

min_year = int(yearly_df['연도'].min())
max_year = int(yearly_df['연도'].max())

# 누락된 연도를 NaN으로 남겨서 그래프 선이 끊어지도록 전체 연도 데이터 병합
all_years = pd.DataFrame({'연도': range(min_year, max_year + 1)})
yearly_df = pd.merge(all_years, yearly_df, on='연도', how='left')

# 선형 회귀 추세선 계산 (결측치 제외 후 전체 기간 기온 상승 폭 측정)
valid_years = yearly_df.dropna(subset=['평균기온'])
slope, intercept = np.polyfit(valid_years['연도'], valid_years['평균기온'], 1)
yearly_df['추세선'] = slope * yearly_df['연도'] + intercept

st.sidebar.image("https://img.icons8.com/fluency/96/thermometer.png", width=64)
st.sidebar.title("🎛️ 분석 옵션 설정")
st.sidebar.markdown("---")

# 연도 범위 슬라이더
selected_years = st.sidebar.slider(
    "📅 조회할 연도 범위",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

# 이동평균 창 크기 선택
ma_window = st.sidebar.selectbox(
    "📈 이동평균선(Smooth Line) 기간 설정",
    options=[3, 5, 10, 15, 20],
    index=2,
    format_func=lambda x: f"{x}년 이동평균"
)

# 보조선 옵션
show_trendline = st.sidebar.checkbox("📐 전체 기간 선형 추세선 표시", value=True)
show_minmax_range = st.sidebar.checkbox("📊 최저/최고 기온 평균 범위 표시", value=False)

# 선택된 연도 데이터 필터링
filtered_yearly = yearly_df[
    (yearly_df['연도'] >= selected_years[0]) & 
    (yearly_df['연도'] <= selected_years[1])
].copy()

# 이동평균 계산
filtered_yearly['이동평균'] = filtered_yearly['평균기온'].rolling(window=ma_window, center=True).mean()

avg_temp_overall = filtered_yearly['평균기온'].mean()
warmest_row = filtered_yearly.loc[filtered_yearly['평균기온'].idxmax()]
coldest_row = filtered_yearly.loc[filtered_yearly['평균기온'].idxmin()]

# 전체 기간 기준 상승 온도 계산
total_warming = slope * (max_year - min_year)
decade_warming = slope * 10

st.markdown('<div class="main-title">🌡️ 서울 100년 연평균 기온 변화 분석</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">1907년부터 최근까지 서울 기상 관측 데이터로 확인하는 한반도 기온 상승 추세 ({min_year}년 ~ {max_year}년)</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">선택 기간 연평균 기온</div>
        <div class="metric-value">{avg_temp_overall:.2f} ℃</div>
        <div class="metric-sub" style="color:#64748B;">총 {filtered_yearly['평균기온'].count()}개 연도 관측</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">가장 무더웠던 해</div>
        <div class="metric-value">{int(warmest_row['연도'])}년</div>
        <div class="metric-sub">연평균 {warmest_row['평균기온']:.2f} ℃</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">가장 추웠던 해</div>
        <div class="metric-value">{int(coldest_row['연도'])}년</div>
        <div class="metric-sub" style="color:#2563EB;">연평균 {coldest_row['평균기온']:.2f} ℃</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">100년당 기온 상승 폭</div>
        <div class="metric-value">+{total_warming:.2f} ℃</div>
        <div class="metric-sub">10년당 약 +{decade_warming:.2f} ℃ 상승</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

st.subheader("📈 연평균 기온 장기 추이 그래프")

fig = go.Figure()

# 최저/최고 기온 범위 표시 (선택 시)
if show_minmax_range:
    fig.add_trace(go.Scatter(
        x=filtered_yearly['연도'],
        y=filtered_yearly['최고기온평균'],
        mode='lines',
        line=dict(width=0),
        connectgaps=False,
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=filtered_yearly['연도'],
        y=filtered_yearly['최저기온평균'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(239, 68, 68, 0.1)',
        name='최저~최고 평균 범위',
        connectgaps=False,
        hoverinfo='skip'
    ))

# 연평균 기온 기본 선 그래프 (누락 연도는 선이 끊기도록 connectgaps=False 설정)
fig.add_trace(go.Scatter(
    x=filtered_yearly['연도'],
    y=filtered_yearly['평균기온'],
    mode='lines+markers',
    name='연평균 기온',
    connectgaps=False,
    line=dict(color='#3B82F6', width=1.5),
    marker=dict(size=5, color='#1D4ED8'),
    hovertemplate='%{x}년: <b>%{y:.2f} ℃</b><extra></extra>'
))

# 이동평균선
fig.add_trace(go.Scatter(
    x=filtered_yearly['연도'],
    y=filtered_yearly['이동평균'],
    mode='lines',
    name=f'{ma_window}년 이동평균',
    connectgaps=False,
    line=dict(color='#EF4444', width=3),
    hovertemplate='%{x}년근처 (%{text}): <b>%{y:.2f} ℃</b><extra></extra>',
    text=[f'{ma_window}년 평균' for _ in range(len(filtered_yearly))]
))

# 선형 추세선 (선택 시)
if show_trendline:
    fig.add_trace(go.Scatter(
        x=filtered_yearly['연도'],
        y=filtered_yearly['추세선'],
        mode='lines',
        name='장기 온난화 추세선',
        line=dict(color='#F59E0B', width=2, dash='dash'),
        hovertemplate='%{x}년 추세선: <b>%{y:.2f} ℃</b><extra></extra>'
    ))

fig.update_layout(
    xaxis_title="연도 (Year)",
    yaxis_title="기온 (℃)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=20, r=20, t=30, b=20),
    height=480,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 10년 단위(시대별) 평균 기온", 
    "🌡️ 일별 기온 분포 (히스토그램)", 
    "🔥 최저 vs 최고기온 (산점도)", 
    "📋 세부 데이터 확인 및 다운로드"
])

with tab1:
    st.subheader("🏛️ 10년 단위(Decade) 평균 기온 변화")
    
    # 10년 단위 그룹화
    yearly_df['시대'] = (yearly_df['연도'] // 10) * 10
    decade_df = yearly_df.groupby('시대').agg(
        시대평균기온=('평균기온', 'mean'),
        연도수=('연도', 'count')
    ).reset_index()
    
    decade_df['시대라벨'] = decade_df['시대'].astype(str) + "년대"
    
    fig_decade = px.bar(
        decade_df,
        x='시대라벨',
        y='시대평균기온',
        text_auto='.2f',
        color='시대평균기온',
        color_continuous_scale='Reds',
        labels={'시대라벨': '시대', '시대평균기온': '평균기온 (℃)'}
    )
    
    fig_decade.update_layout(
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        yaxis=dict(range=[decade_df['시대평균기온'].min() - 0.5, decade_df['시대평균기온'].max() + 0.5])
    )
    
    st.plotly_chart(fig_decade, use_container_width=True)
    st.caption("💡 최근 시대로 올수록 서울의 연평균 기온이 지속적으로 상승하는 명확한 온난화 경향을 보입니다.")

with tab2:
    st.subheader("🌡️ 선택 기간 일별 평균기온 분포 (히스토그램)")
    
    # 선택된 기간의 일별 데이터 필터링
    filtered_daily = df_raw[
        (df_raw['연도'] >= selected_years[0]) & 
        (df_raw['연도'] <= selected_years[1])
    ].copy()
    
    # 히스토그램 차트 생성
    fig_hist = px.histogram(
        filtered_daily,
        x='평균기온',
        nbins=50,
        labels={'평균기온': '일 평균기온 (℃)', 'count': '일수 (일)'},
        color_discrete_sequence=['#3B82F6'],
        opacity=0.85
    )
    
    # 평균기온 및 중앙값 표시 선 추가
    mean_temp = filtered_daily['평균기온'].mean()
    median_temp = filtered_daily['평균기온'].median()
    
    fig_hist.add_vline(
        x=mean_temp, 
        line_dash="dash", 
        line_color="#EF4444", 
        annotation_text=f"평균: {mean_temp:.1f}℃", 
        annotation_position="top left"
    )
    fig_hist.add_vline(
        x=median_temp, 
        line_dash="dot", 
        line_color="#10B981", 
        annotation_text=f"중앙값: {median_temp:.1f}℃", 
        annotation_position="top right"
    )
    
    fig_hist.update_layout(
        yaxis_title="날짜 수 (일)",
        xaxis_title="일 평균기온 (℃)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        bargap=0.08
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("💡 선택한 연도 구간의 일별 평균기온 분포를 보여줍니다. 평균값(빨간 점선)과 중앙값(초록 점선)을 통해 기온이 어느 온도 구간에 가장 집중되어 있는지 한눈에 확인할 수 있습니다.")

with tab3:
    st.subheader("🔥 일별 최저기온 vs 최고기온 상관관계 (산점도)")
    
    # 선택된 기간 일별 데이터 (결측치 제거)
    filtered_daily = df_raw[
        (df_raw['연도'] >= selected_years[0]) & 
        (df_raw['연도'] <= selected_years[1])
    ].copy()
    
    scatter_df = filtered_daily.dropna(subset=['최저기온', '최고기온']).copy()
    
    # 상관계수 계산
    corr = scatter_df['최저기온'].corr(scatter_df['최고기온'])
    
    st.markdown(f"**선택 기간 최저기온과 최고기온의 피어슨 상관계수:** `{corr:.3f}` (매우 강한 양의 상관관계)")
    
    # 산점도 그래프 생성 (대용량 데이터 속도를 위해 webgl 활용)
    fig_scatter = px.scatter(
        scatter_df,
        x='최저기온',
        y='최고기온',
        color='월',
        color_continuous_scale='Turbo',
        opacity=0.5,
        labels={'최저기온': '일 최저기온 (℃)', '최고기온': '일 최고기온 (℃)', '월': '월(Month)'},
        hover_data={'날짜': '|%Y-%m-%d', '평균기온': ':.1f'},
        render_mode='webgl'
    )
    
    # y = x 대각선 기준선 (일교차가 0인 선) 추가
    min_val = min(scatter_df['최저기온'].min(), scatter_df['최고기온'].min()) - 2
    max_val = max(scatter_df['최저기온'].max(), scatter_df['최고기온'].max()) + 2
    
    fig_scatter.add_shape(
        type="line",
        x0=min_val, y0=min_val,
        x1=max_val, y1=max_val,
        line=dict(color="Gray", width=1.5, dash="dash"),
    )
    
    fig_scatter.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        coloraxis_colorbar=dict(title="월 (Month)", dtick=1)
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("💡 점의 색상은 해당 날짜의 월(1~12월)을 나타냅니다. 회색 점선은 최저기온과 최고기온이 동일한 선(일교차 0℃)으로, 점이 이 선에서 수직으로 멀어질수록 그날의 일교차가 크다는 것을 보여줍니다.")

with tab4:
    st.subheader("📋 선택 기간 연도별 통계 데이터")
    
    # 관측 데이터가 있는 연도만 표에 표시
    display_df = filtered_yearly.dropna(subset=['평균기온'])[['연도', '평균기온', '최저기온평균', '최고기온평균', '최고기온극값', '최저기온극값']].copy()


with tab1:
    st.subheader("🏛️ 10년 단위(Decade) 평균 기온 변화")
    
    # 10년 단위 그룹화
    yearly_df['시대'] = (yearly_df['연도'] // 10) * 10
    decade_df = yearly_df.groupby('시대').agg(
        시대평균기온=('평균기온', 'mean'),
        연도수=('연도', 'count')
    ).reset_index()
    
    decade_df['시대라벨'] = decade_df['시대'].astype(str) + "년대"
    
    fig_decade = px.bar(
        decade_df,
        x='시대라벨',
        y='시대평균기온',
        text_auto='.2f',
        color='시대평균기온',
        color_continuous_scale='Reds',
        labels={'시대라벨': '시대', '시대평균기온': '평균기온 (℃)'}
    )
    
    fig_decade.update_layout(
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        yaxis=dict(range=[decade_df['시대평균기온'].min() - 0.5, decade_df['시대평균기온'].max() + 0.5])
    )
    
    st.plotly_chart(fig_decade, use_container_width=True)
    st.caption("💡 최근 시대로 올수록 서울의 연평균 기온이 지속적으로 상승하는 명확한 온난화 경향을 보입니다.")

with tab2:
    st.subheader("🌡️ 선택 기간 일별 평균기온 분포 (히스토그램)")
    
    # 선택된 기간의 일별 데이터 필터링
    filtered_daily = df_raw[
        (df_raw['연도'] >= selected_years[0]) & 
        (df_raw['연도'] <= selected_years[1])
    ].copy()
    
    # 히스토그램 차트 생성
    fig_hist = px.histogram(
        filtered_daily,
        x='평균기온',
        nbins=50,
        labels={'평균기온': '일 평균기온 (℃)', 'count': '일수 (일)'},
        color_discrete_sequence=['#3B82F6'],
        opacity=0.85
    )
    
    # 평균기온 및 중앙값 표시 선 추가
    mean_temp = filtered_daily['평균기온'].mean()
    median_temp = filtered_daily['평균기온'].median()
    
    fig_hist.add_vline(
        x=mean_temp, 
        line_dash="dash", 
        line_color="#EF4444", 
        annotation_text=f"평균: {mean_temp:.1f}℃", 
        annotation_position="top left"
    )
    fig_hist.add_vline(
        x=median_temp, 
        line_dash="dot", 
        line_color="#10B981", 
        annotation_text=f"중앙값: {median_temp:.1f}℃", 
        annotation_position="top right"
    )
    
    fig_hist.update_layout(
        yaxis_title="날짜 수 (일)",
        xaxis_title="일 평균기온 (℃)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly_white",
        bargap=0.08
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("💡 선택한 연도 구간의 일별 평균기온 분포를 보여줍니다. 평균값(빨간 점선)과 중앙값(초록 점선)을 통해 기온이 어느 온도 구간에 가장 집중되어 있는지 한눈에 확인할 수 있습니다.")

with tab3:
    st.subheader("📋 선택 기간 연도별 통계 데이터")
    
    # 관측 데이터가 있는 연도만 표에 표시
    display_df = filtered_yearly.dropna(subset=['평균기온'])[['연도', '평균기온', '최저기온평균', '최고기온평균', '최고기온극값', '최저기온극값']].copy()
    display_df.columns = ['연도', '연평균기온(℃)', '최저기온평균(℃)', '최고기온평균(℃)', '연중최고기온(℃)', '연중최저기온(℃)']
    
    st.dataframe(
        display_df.style.format({
            '연평균기온(℃)': '{:.2f}',
            '최저기온평균(℃)': '{:.2f}',
            '최고기온평균(℃)': '{:.2f}',
            '연중최고기온(℃)': '{:.1f}',
            '연중최저기온(℃)': '{:.1f}'
        }),
        use_container_width=True,
        height=350
    )
    
    csv_data = display_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 CSV 데이터 다운로드",
        data=csv_data,
        file_name="seoul_yearly_temperature.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>데이터 출처: 기상청 공공데이터 포털 / modudata repositorio (seoul.csv)</p>", unsafe_allow_html=True)
