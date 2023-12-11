import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from model import load_model_and_predict

DATA = 'data/df_client_agg.csv'


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


df_client_agg = load_data()

st.title("""Классификация клиентов банка""")
st.write("""Определяем кто из клиентов откликнется на предложение банка""")



st.write("""__1. EDA. Распределения признаков__""")
fig, axes = plt.subplots(5, 2, figsize=(25, 20))

axes[0][0].hist(df_client_agg['AGE'])
axes[0][0].set_title('Distribution by AGE')

axes[0][1].hist(df_client_agg['GENDER'])
axes[0][1].set_title('Distribution by GENDER')

axes[1][0].hist(df_client_agg['CHILD_TOTAL'])
axes[1][0].set_title('Distribution by CHILD_TOTAL')

axes[1][1].hist(df_client_agg['DEPENDANTS'])
axes[1][1].set_title('Distribution by DEPENDANTS')

axes[2][0].hist(df_client_agg['SOCSTATUS_WORK_FL'])
axes[2][0].set_title('Distribution by SOCSTATUS_WORK_FL')

axes[2][1].hist(df_client_agg['SOCSTATUS_PENS_FL'])
axes[2][1].set_title('Distribution by SOCSTATUS_PENS_FL')

axes[3][0].hist(df_client_agg['PERSONAL_INCOME'])
axes[3][0].set_title('Distribution by PERSONAL_INCOME')

axes[3][1].hist(df_client_agg['LOAN_NUM_TOTAL'])
axes[3][1].set_title('Distribution by LOAN_NUM_TOTAL')

axes[4][0].hist(df_client_agg['LOAN_NUM_CLOSED'])
axes[4][0].set_title('Distribution by LOAN_NUM_CLOSED')

axes[4][1].hist(df_client_agg['TARGET'])
axes[4][1].set_title('Distribution by TARGET')

st.pyplot(fig)

features = st.sidebar.multiselect(
    '__Выберите переменные для построения корреляционной матрицы__',
    df_client_agg.iloc[:, 2:].columns
)



features2 = st.sidebar.multiselect(
    '__Выберите переменные для расчета основных статистик__',
    df_client_agg.iloc[:, 2:].columns
)

if len(features) == 0:
    st.write("""__2. Выберите переменные для корреляционной матрицы в меню слева__""")
else:
    st.write("""__2. EDA. Корреляционная матрица__""")
    st.dataframe(df_client_agg[features].corr())

if len(features) == 0:
    st.write("""__3. Для построения тепловой карты выберите переменные для корреляционной матрицы в меню слева__""")
else:
    st.write("""__3. EDA. Тепловая карта корреляционной матрицы__""")
    fig, ax = plt.subplots()
    sns.heatmap(df_client_agg[features].corr(), ax=ax, cmap='Spectral')
    st.write(fig)

if len(features2) == 0:
    st.write("""__4. Для вычисления основных статистик выберите переменные в меню слева__""")
else:
    st.write("""__4. EDA. Вычисление числовых характеристик распределения числовых столбцов__""")
    st.dataframe(df_client_agg[features2].describe())

st.write("""__5. Данные пользователя__""")
def sidebar_input_features():
    AGE = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
                            step=1)
    SOCSTATUS_WORK_FL = st.sidebar.selectbox("Работаете ли вы?", ("Да", "Нет"))
    SOCSTATUS_PENS_FL = st.sidebar.selectbox("Являетесь ли вы пенсионером?", ("Да", "Нет"))
    GENDER = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    CHILD_TOTAL = st.sidebar.slider("Кол-во детей", min_value=0, max_value=20, value=0,
                            step=1)
    DEPENDANTS = st.sidebar.slider("Кол-во иждивенцев", min_value=0, max_value=20, value=0,
                                    step=1)
    PERSONAL_INCOME = st.sidebar.slider("Личный доход (тыс. руб./мес.)", min_value=0, max_value=1000, value=100,
                                    step=10)
    LOAN_NUM_TOTAL = st.sidebar.slider("Кол-во ссуд", min_value=0, max_value=20, value=0,
                                    step=1)
    LOAN_NUM_CLOSED = st.sidebar.slider("Кол-во погашенных ссуд", min_value=0, max_value=20, value=0,
                                       step=1)
    translation = {
        "Мужской": 1,
        "Женский": 0,
        "Да": 1,
        "Нет": 0
    }

    data = {
        "AGE": AGE,
        "GENDER": translation[GENDER],
        "CHILD_TOTAL": CHILD_TOTAL,
        "DEPENDANTS": DEPENDANTS,
        "SOCSTATUS_WORK_FL": translation[SOCSTATUS_WORK_FL],
        "SOCSTATUS_PENS_FL": translation[SOCSTATUS_PENS_FL],
        "PERSONAL_INCOME": PERSONAL_INCOME,
        "LOAN_NUM_TOTAL": LOAN_NUM_TOTAL,
        "LOAN_NUM_CLOSED": LOAN_NUM_CLOSED
    }

    df = pd.DataFrame(data, index=[0])
    st.dataframe(df)
    return df

def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    prediction, prediction_probas = load_model_and_predict(user_input_df)
    write_prediction(prediction, prediction_probas)

if __name__ == "__main__":
    process_side_bar_inputs()

