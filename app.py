import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Student Performance Analytics", page_icon="🎓", layout="wide")
REQ=["Gender","EthnicGroup","ParentEduc","LunchType","TestPrep","ParentMaritalStatus","PracticeSport","IsFirstChild","NrSiblings","TransportMeans","WklyStudyHours","MathScore","ReadingScore","WritingScore"]
SCORES=["MathScore","ReadingScore","WritingScore"]

def demo_data(n=300):
    rng=np.random.default_rng(42)
    p=rng.choice(["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"],n,p=[.13,.20,.22,.18,.18,.09])
    prep=rng.choice(["none","completed"],n,p=[.68,.32]); study=rng.choice(["< 5","5 - 10","> 10"],n,p=[.35,.48,.17]); lunch=rng.choice(["standard","free/reduced"],n,p=[.65,.35])
    base=58+pd.Series(p).map({"some high school":0,"high school":3,"some college":6,"associate's degree":8,"bachelor's degree":11,"master's degree":14}).to_numpy()+pd.Series(prep).map({"none":0,"completed":7}).to_numpy()+pd.Series(study).map({"< 5":0,"5 - 10":5,"> 10":9}).to_numpy()+pd.Series(lunch).map({"standard":4,"free/reduced":0}).to_numpy()+rng.normal(0,8,n)
    base=np.clip(base,0,100)
    return pd.DataFrame({"Gender":rng.choice(["female","male"],n),"EthnicGroup":rng.choice(["group A","group B","group C","group D","group E"],n),"ParentEduc":p,"LunchType":lunch,"TestPrep":prep,"ParentMaritalStatus":rng.choice(["married","single","widowed","divorced"],n),"PracticeSport":rng.choice(["never","sometimes","regularly"],n),"IsFirstChild":rng.choice(["yes","no"],n),"NrSiblings":rng.integers(0,6,n),"TransportMeans":rng.choice(["school_bus","private"],n),"WklyStudyHours":study,"MathScore":np.clip(base+rng.normal(0,6,n),0,100).round(),"ReadingScore":np.clip(base+4+rng.normal(0,6,n),0,100).round(),"WritingScore":np.clip(base+5+rng.normal(0,6,n),0,100).round()})

def clean(df):
    df=df.copy()
    df=df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],errors="ignore")
    for c in df.columns:
        if df[c].dtype=="object": df[c]=df[c].replace(r"^\s*$",np.nan,regex=True)
    if "NrSiblings" in df: df["NrSiblings"]=pd.to_numeric(df["NrSiblings"],errors="coerce")
    for c in SCORES: df[c]=pd.to_numeric(df[c],errors="coerce")
    for c in df.select_dtypes("object"): df[c]=df[c].fillna("Unknown")
    for c in df.select_dtypes(include=np.number): df[c]=df[c].fillna(df[c].median())
    df["AverageScore"]=df[SCORES].mean(axis=1)
    df["PerformanceBand"]=pd.cut(df["AverageScore"],[-np.inf,49.99,69.99,84.99,np.inf],labels=["Needs Support","Developing","Strong","Outstanding"])
    return df

st.title("🎓 Student Performance Analytics")
st.caption("Recruiter-ready EDA dashboard • Python • Pandas • NumPy • Plotly • Scikit-learn")
with st.sidebar:
    upload=st.file_uploader("Upload Expanded_data_with_more_features.csv",type="csv")
    demo=st.checkbox("Use synthetic demo data",value=upload is None)
    st.markdown("**Question:** Which demographic, family, resource and study-related factors are associated with student performance?")
if upload and not demo:
    raw=pd.read_csv(upload); source="Uploaded dataset"
else:
    raw=demo_data(); source="Synthetic demo"
missing=[c for c in REQ if c not in raw.columns]
if missing and source=="Uploaded dataset": st.error("Missing required columns: "+", ".join(missing)); st.stop()
df=clean(raw)

a,b,c,d=st.columns(4); a.metric("Students",f"{len(df):,}"); b.metric("Average score",f"{df.AverageScore.mean():.1f}"); c.metric("Highest score",f"{df[SCORES].max().max():.0f}"); d.metric("Columns",len(df.columns))
t1,t2,t3,t4,t5=st.tabs(["📊 Overview","🔎 EDA","🧩 Relationships","🤖 Baseline Model","📋 Data Quality"])

with t1:
    sm=df[SCORES].mean().rename({"MathScore":"Math","ReadingScore":"Reading","WritingScore":"Writing"}).reset_index(); sm.columns=["Subject","Average"]
    st.plotly_chart(px.bar(sm,x="Subject",y="Average",text_auto=".1f",title="Average score by subject").update_yaxes(range=[0,100]),use_container_width=True)
    st.info("The original project focuses on association/pattern discovery, not causation. This version adds preprocessing, feature engineering, a baseline model and an interactive dashboard.")

with t2:
    f=st.selectbox("Group by",["ParentEduc","TestPrep","LunchType","Gender","EthnicGroup","WklyStudyHours","PracticeSport","ParentMaritalStatus","TransportMeans"])
    m=st.selectbox("Score",SCORES+["AverageScore"],index=3)
    g=df.groupby(f)[m].agg(["mean","count"]).reset_index().sort_values("mean",ascending=False)
    st.plotly_chart(px.bar(g,x=f,y="mean",text_auto=".1f",hover_data=["count"],title=f"{m} by {f}").update_yaxes(range=[0,100]),use_container_width=True)
    x,y=st.columns(2)
    with x: st.plotly_chart(px.histogram(df,x="AverageScore",nbins=20,title="Average-score distribution"),use_container_width=True)
    with y: st.plotly_chart(px.box(df,x="TestPrep",y="AverageScore",title="Performance by test preparation"),use_container_width=True)

with t3:
    num=df[SCORES+["NrSiblings","AverageScore"]].select_dtypes("number")
    st.plotly_chart(px.imshow(num.corr(),text_auto=".2f",aspect="auto",title="Numeric correlation matrix"),use_container_width=True)
    x=st.selectbox("X-axis",["WklyStudyHours","NrSiblings","ParentEduc","TestPrep","LunchType"]); y=st.selectbox("Y-axis",SCORES)
    if pd.api.types.is_numeric_dtype(df[x]): fig=px.scatter(df,x=x,y=y,trendline="ols",title=f"{y} vs {x}")
    else: fig=px.box(df,x=x,y=y,points=False,title=f"{y} across {x}")
    st.plotly_chart(fig,use_container_width=True)

with t4:
    st.write("Ridge regression baseline predicting AverageScore. This extends the original project's stated future scope of predictive modeling.")
    features=[c for c in REQ if c not in SCORES and c in df.columns]; X=df[features]; yy=df.AverageScore
    cat=X.select_dtypes(include=["object","category"]).columns.tolist(); num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    model=Pipeline([("preprocess",pre),("model",Ridge(alpha=1.0))])
    Xtr,Xte,ytr,yte=train_test_split(X,yy,test_size=.2,random_state=42); model.fit(Xtr,ytr); pred=model.predict(Xte)
    m1,m2=st.columns(2); m1.metric("MAE",f"{mean_absolute_error(yte,pred):.2f}"); m2.metric("R²",f"{r2_score(yte,pred):.3f}")
    ev=pd.DataFrame({"Actual":yte,"Predicted":pred}); st.plotly_chart(px.scatter(ev,x="Actual",y="Predicted",title="Actual vs predicted AverageScore"),use_container_width=True)

with t5:
    q=pd.DataFrame({"Column":raw.columns,"Missing":raw.isna().sum().values,"Missing %":(raw.isna().mean().values*100).round(2),"Unique":raw.nunique(dropna=True).values,"Type":raw.dtypes.astype(str).values})
    st.dataframe(q,use_container_width=True,hide_index=True)
    st.write("Cleaning: blank categorical cells → missing; categorical gaps → Unknown; numeric gaps → median; AverageScore and PerformanceBand are engineered.")
st.caption(f"Source mode: {source}")
