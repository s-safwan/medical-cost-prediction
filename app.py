"""Optional local demo: streamlit run app.py."""
import joblib
import streamlit as st
from model import ROOT, CATEGORIES, validate_record
st.set_page_config(page_title='Medical Cost Explorer', page_icon='📊')
st.title('Medical Cost Explorer')
st.caption('An educational regression demo by Syed Safwan Hashmi')
st.info('Estimates dataset medical charges. This is not an insurance premium, medical recommendation, or validated financial tool.')
@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')
if not (ROOT / 'model.joblib').exists():
    st.warning('Run python train.py first, then restart this app.')
    st.stop()
bundle = load_model()
with st.form('prediction'):
    a,b = st.columns(2)
    age = a.number_input('Age',18,64,30)
    bmi = b.number_input('BMI',15.0,54.0,25.0,step=0.1)
    children = a.number_input('Children',0,5,0)
    gender = b.selectbox('Gender (dataset field)', CATEGORIES['gender'])
    smoker = a.selectbox('Smoker', CATEGORIES['smoker'])
    region = b.selectbox('Region (US dataset categories)', CATEGORIES['region'])
    submit = st.form_submit_button('Estimate annual charges')
if submit:
    frame = validate_record(dict(age=age,bmi=bmi,children=children,gender=gender,smoker=smoker,region=region))
    value = float(bundle['model'].predict(frame)[0])
    st.metric('Estimated annual charges (dataset units)', f'{value:,.0f}')
    st.caption('A point estimate with substantial uncertainty; not a quote. No user inputs are saved by this app.')
m = bundle['metadata']['metrics']
st.subheader('Held-out evaluation')
a,b,c=st.columns(3)
a.metric('MAE',f"{m['test_mae']:,.0f}"); b.metric('RMSE',f"{m['test_rmse']:,.0f}"); c.metric('R²',f"{m['test_r2']:.3f}")
st.caption(f"Selected model: {m['selected_model']}. Model choice used training-only cross-validation.")
st.image(str(ROOT/'reports/diagnostics.png'))
st.caption('Small dataset; source and license require confirmation. Region and demographic associations do not establish causation or fairness.')
