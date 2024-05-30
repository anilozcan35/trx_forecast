import os
import warnings
import requests

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components



def home():
    """Home Page of Streamlit UI"""
    st.title('TRX Forecast')
    st.subheader('Proje kapsamında 3 model denenmiştir.')
    st.markdown('**1. System Design and Pipeline')
    st.markdown('**2. LSTM Prediction Section')
    st.markdown('**3. Lightgbm Prediction Section')
    st.markdown('**4. Smoothing')

def pipeline():
    """Project System Design Page"""
    st.title('Project System Design')
    st.image(image="./pipelineDesign/systemdesign.png",
             caption="Project System Design",
             #  width=200,
             use_column_width="auto"
             )


def ligbhtgbm():
    """Light GBM Prediction Page"""
    st.subheader('LightGbm Prediction Page')

def lstm():
    """LSTM Prediction Page"""
    st.subheader('LSTM Prediction Page')

    tab1, tab2 = st.tabs(["Bulk Prediction", "Unit Prediction"])

    with tab1:
        CURRENT_DIR = os.getcwd()
        TEST_PATH = os.path.join(CURRENT_DIR, "data\\test.csv")
        print(TEST_PATH)
        test_df = pd.read_csv(TEST_PATH, index_col=0)
        st.dataframe(test_df)
        lstm_predict = st.button("Predict LSTM")
        if lstm_predict:
            response = requests.post("http://localhost:5001/predict_lstm", json={"trigger": True})
            st.markdown(response)




def smoothing():
    """Smoothing Prediction Page"""
    st.subheader('Smoothing Prediction Page')

    tab1, tab2 = st.tabs(["Bulk Prediction", "Unit Prediction"])

    with tab1:
        CURRENT_DIR = os.getcwd()
        TEST_PATH = os.path.join(CURRENT_DIR, "data\\test.csv")
        print(TEST_PATH)
        test_df = pd.read_csv(TEST_PATH, index_col=0)
        st.dataframe(test_df)
        smoothing_predict = st.button("Predict Smoothing")
        if smoothing_predict: # Smoothing butonu tıklanırsa api istek gönderilir.
            response = requests.post("http://localhost:5001/predict_smoothing", json={"trigger": True})
            st.markdown(response)




def app_credits():
    """App Info. & Credits Page"""
    st.title('App Info. & Credits')
    st.markdown('**Programming Language:** Python 3.12')
    st.markdown('**Libraries & Frameworks:** Pandas, Scikit-learn, Numpy, Matplotlib, Seaborn, Plotly, Yellowbrick')
    st.markdown('**UI:** [Streamlit](https://streamlit.io/)')
    st.markdown('**Dev. Tools:** Git, Docker')
    st.markdown('**Data Source:** [Kaggle](https://www.kaggle.com/kukuroo3/body-performance-data)')
    st.markdown('**Github Repo:** [Web Mining Project](https://github.com/)')
    st.markdown('**Dash Platform:** [Streamlit Community Cloud](https://streamlit.io/cloud)')
    st.markdown('**Dash Url:** [StreamLit App](https://web-mining-project.streamlit.app/)')
    st.markdown(
        '**Developed by:** [Metin Uslu](http://linkedin.com/in/metinuslu) & [Anıl Özcan](https://www.linkedin.com/in/anil-ozcan-6ba16b152/)')


def get_menu(user_name="local", user_password="local"):
    """Streamlit UI Menu
    Params:
        user_name: str
        user_password: str
    """

    # st.sidebar.image("static/sidebar_logo.png")
    # , use_column_width=True
    st.sidebar.title('TRX Forecast')
    side_menu = {
        'Home': home,
        'System Design': pipeline,
        'Lightgbm': ligbhtgbm,
        'LSTM': lstm,
        'Smoothing': smoothing,
    }

    if st.session_state.get('login_success'):
        choice = st.sidebar.radio('Applications', list(side_menu.keys()))
        side_menu[choice]()
    else:
        with st.sidebar:
            with st.form(key='login_form'):
                st.title('Loging Page')
                username = st.text_input('User Name')
                password = st.text_input('Password', type='password')
                if st.form_submit_button('Login'):
                    if username == user_name and password == user_password:
                        st.session_state['login_success'] = True
                        st.success('Giriş başarılı, yönlendiriliyorsunuz...')
                        st.experimental_rerun()
                    else:
                        st.error('Kullanıcı adı veya şifre yanlış.')
                        st.session_state['login_success'] = False
    # show_pages_from_config()


if __name__ == "__main__":
    # Set Constants
    ROOT_PATH = os.getcwd()

    st.set_page_config(
        page_title="Trx Forecast UI ",
        page_icon=":gem:",
        layout="wide",
        # layout="centered",
        initial_sidebar_state="expanded",
        # initial_sidebar_state="auto",
        # menu_items=None,
        menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
                    'Report a bug': "https://www.extremelycoolapp.com/bug",
                    'About': "# This is a header. This is an *extremely* cool app!"
                    })

    # Streamlit Menu
    get_menu()
