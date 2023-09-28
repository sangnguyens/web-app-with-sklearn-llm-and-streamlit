#!/usr/bin/env python
import streamlit as st
import pandas as pd
import joblib as jl
import tqdm

from skllm.config import SKLLMConfig
from skllm import ZeroShotGPTClassifier
from skllm.datasets import get_classification_dataset
import os
from dotenv import load_dotenv

# instatiate env
load_dotenv()

# set openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ORGANIZATION_NAME = os.getenv("ORGANIZATION_NAME")

# set config
SKLLMConfig.set_openai_key(OPENAI_API_KEY)
# SKLLMConfig.set_openai_org(ORGANIZATION_NAME)


#################### Build model ##########################
def build_model():
    # get data
    X, y = get_classification_dataset()

    # define model
    clf = ZeroShotGPTClassifier(openai_model="gpt-3.5-turbo")

    # train model
    clf.fit(None, ["positive", "negative", "neutral"])
    
    # save model
    os.makedirs("saved_model", exist_ok=True)
    jl.dump(clf, "saved_model/model.joblib")

#################### Streamlit App ###########################################
def run_app(model_path, *args):
    # set layout and title
    st.set_page_config(layout="wide", page_title="Sentiment Analysis Application")
    st.title("Sentiment Classifier App using Scikit Learn LLM")

    # upload file
    upload = st.file_uploader("Upload your CSV file", type=["csv"])

    # load model
    clf = jl.load(model_path)

    # get data from upload file
    if upload is not None:
        if st.button("Analyze csv File"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.info("CSV File uploaded")
                csv_file = upload.name
                with open(os.path.join(csv_file), "wb") as f:
                    f.write(upload.getbuffer())
                print(csv_file)

                df = pd.read_csv(csv_file, encoding="unicode_escape", index_col=None)
                st.dataframe(df, use_container_width=True)
            with col2:
                data_list = df["Review"].tolist()
                labels = clf.predict(data_list)
                df["Sentiment"] = labels
                st.info("Sentiment Analysis Result")
                st.dataframe(df, use_container_width=True)
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download data as CSV",
                    data=csv_data,
                    file_name="result_df.csv",
                    mime="text/csv",
                )

def main(saved_model="saved_model/model.joblib"):
    try:
        clf = jl.load(saved_model)
        print("Loading model from disk")
    except:
        print("Build model...")
        build_model()
    run_app(saved_model)


if __name__ == "__main__":
    main()
