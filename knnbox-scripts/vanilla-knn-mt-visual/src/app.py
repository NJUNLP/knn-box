import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt


from function import get_icon, get_spatial_distribution, \
    get_value_frequency, display_partial_records, get_knn_model_resource, translate_using_knn_model, \
        get_config, get_datastore_cfgs \

from tokenizer import TOKENIZER_FUNCTIONS

APP_TITLE = "knn-playground"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# main function
def knn_main():
    st.sidebar.image(get_icon(), use_column_width=True)
    operation_type = st.sidebar.selectbox("Select function:", ("Translation", "Datastore Profiling"))


    with st.sidebar:
        with st.expander("⚡ About this app", expanded=True):
            st.write(
        """     
-   Powered by [knn-box](https://github.com/NJUNLP/knn-box) toolkit.
-   Developed by Nanjing University NLP Group.
	    """
    )

        st.markdown("")

    # show datastore infomations
    if operation_type == "Datastore Profiling":
        with st.form("ds_path"):
            path_of_datasotre = st.text_input(
                label="Paste the path of datastore below",
                value="/home/demo/knn-box/datastore/vanilla-visual/zh-en-laws"
            )
            path_of_dictionary = st.text_input(
                label="Paste the path of target language dictionary",
                value="/home/demo/knn-box/data-bin/zh-en-laws/dict.zh.txt",
            )
            path_submit = st.form_submit_button(label="✨ Get me the profile!")
        
        if "has_clicked" not in st.session_state:
            st.session_state.has_clicked = False
        if "last_ds_path" not in st.session_state:
            st.session_state.last_ds_path = ""
        if "last_dic_path" not in st.session_state:
            st.session_state.last_dic_path = ""

        # we use session here to display the `value frequency chart` when slider changes.
        if path_submit or (st.session_state.has_clicked and 
            st.session_state.last_ds_path == path_of_datasotre and 
            st.session_state.last_dic_path == path_of_dictionary):

            st.session_state.has_clicked = True
            st.session_state.last_ds_path = path_of_datasotre
            st.session_state.last_dic_path = path_of_dictionary
            # progress bar
            with st.spinner('Wait for it...'):
                records = get_value_frequency(path_of_datasotre, path_of_dictionary) 
                datastore_cfgs = get_datastore_cfgs(path_of_datasotre)
                datastore_entries_size = datastore_cfgs["data_infos"]["keys"]["shape"][0]
                datastore_word_size = len(records)
            st.success("Done")

            cols = st.columns(2)
            with cols[0]:
                st.metric("Datastore Entries", datastore_entries_size)
                st.title("Spatial Distribution")
                sample_sz = st.slider("# sample size", min_value=100, max_value=min(100000,datastore_entries_size),
                    value = 30000, step=10000, help="sample how many entries to display",
                )
                chart = get_spatial_distribution(path_of_datasotre, path_of_dictionary, sample_nums=sample_sz)
                st.altair_chart(chart, use_container_width=True)
                st.info("Hint: You can click a point to show a single word's distribution")
            with cols[1]:
                st.metric("Word Counts", datastore_word_size)
                st.title("Word Frequency")
                area = st.slider("# frequency top ratio", min_value=0.00, max_value=0.999, value=0.7, 
                    help="dispaly partial area")
                chart = display_partial_records(records, area, 20)
                st.altair_chart(chart, use_container_width=True)
    

    if operation_type == "Translation":
        cfgs = get_config()
        with st.form(key="my_form"):
            ce, c1, ce, c2, ce = st.columns([0.07, 1, 0.07, 5, 0.07])
            with c1:
                lang_pair = st.selectbox("Choose translation language pair", options=cfgs.keys(),
                help="choose language pair [with domain]",
                )
                
                k = st.number_input(
                    label="K",
                    min_value=1,
                    max_value=64,
                    value=8,
                    help="set the K parameter of kNN-MT"
                )
                lambda_ = st.slider(label="Lambda", min_value=0.00, max_value=1.00, value=0.7, 
                    help="set the Lambda parameter of kNN-MT")
                temperature = st.slider(label="Temperature", min_value=0.01, max_value=100.00, value=10.0, 
                    help="set the Temperature parameter of kNN-MT")
                
            with c2:
                doc = st.text_area("Paste the source language text below (max 500 words)", height=300)
                submit_botton = st.form_submit_button(label="✨ Get me the translation!")
        
        if submit_botton:
            st.title("Translation")
            # prepare model and generator
    
            resource = get_knn_model_resource(**cfgs[lang_pair])
            # choose tokenizer
            doc = TOKENIZER_FUNCTIONS[cfgs[lang_pair]["tokenizer"]](doc)
            nmt_translation_results = translate_using_knn_model(doc, resource,
                    k = 1, lambda_=0.0, temperature=1.0
            )
            translation_results = translate_using_knn_model(doc, resource, 
                k=k, lambda_=lambda_, temperature=temperature)
            nmt_translations = " ".join(nmt_translation_results["hypo_tokens_str"])
            translations = translation_results["hypo_tokens_str"]


            st.subheader("NMT: ")
            st.markdown(nmt_translations)

            st.subheader("kNN-NMT: ")
            tabs = st.tabs(translations)

            for idx, tab in enumerate(tabs):
                with tab:
                    probability_data = pd.DataFrame(
                        {
                            "NMT candidates": translation_results["neural_candis"][idx][:min(k,100)],
                            "NMT probability": translation_results["neural_probs"][idx][:min(k,100)],
                            "kNN-MT candidates": translation_results["combined_candis"][idx][:min(k,100)],
                            "kNN-MT probability": translation_results["combined_probs"][idx][:min(k,100)],
                        },
                        columns=["NMT candidates", "NMT probability", "kNN-MT candidates", "kNN-MT probability"]
                    )
                    df = (
                        probability_data
                    )


                    # Add styling
                    cmGreen = sns.light_palette("green", as_cmap=True)
                    cmRed = sns.light_palette("red", as_cmap=True)
                    df = df.style.background_gradient(
                        cmap=cmGreen,
                        subset=[
                            "NMT probability",
                            "kNN-MT probability",
                        ],
                    )
                    format_dictionary = {
                        "NMT probability": "{:.3f}",
                        "kNN-MT probability": "{:.3f}"
                    }
                    
                    df = df.format(format_dictionary)
                    st.table(df)


                    chart = get_neighbors_chart(translation_results, idx)
                    st.altair_chart(chart,use_container_width=True)

            st.info("1. The distance shown in the graph after PCA does not necessarily align with the actual vector distance, you should refer the actual distance (hover your mouse over the point) \n 2. Becasue of beam search, the token with the highest probability in a single step will not necessarily be chosen")       


def get_neighbors_chart(translation_results, page_idx):

    ds = pd.DataFrame({
        "x": translation_results["knn_neighbors_keys"][page_idx, :, 0],
        "y": translation_results["knn_neighbors_keys"][page_idx, :, 1],
        "value":translation_results["knn_neighbors_values"][page_idx],
        "distance":translation_results["knn_l2_distance"][page_idx],
        "context src": translation_results["knn_context_src"][page_idx],
        "context ref": translation_results["knn_context_ref"][page_idx],
    })

    ds2 = pd.DataFrame({
        "x": translation_results["query_point"][page_idx, 0],
        "y": translation_results["query_point"][page_idx, 1],
        "value":["Query Point"],
        "distance":[0.0],
    })
    chart1 = alt.Chart(ds).mark_circle(size=100).encode(
        x="x",
        y="y",
        color="value",
        tooltip=["value", "context src", "context ref", "distance"],
    ).interactive()

    chart2 = alt.Chart(ds2).mark_circle(size=400).encode(
        x="x",
        y="y",
        color="value",
        tooltip=["x", "y", "distance"],
    ).interactive()
    
    return chart1 + chart2



# Run the app
if __name__ == "__main__":
    knn_main()









