

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Clustering & Recommendation System")
st.write(
    "Find similar Netflix titles using AI-based content clustering."
)


@st.cache_data
def load_data():
    df = pd.read_csv("NetflixSimple.csv")
    return df

df = load_data()


@st.cache_resource
def load_similarity():
    sim = pickle.load(open("similarity.pkl", "rb"))
    return sim

similarity_matrix = load_similarity()


indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def recommend(title, top_n=5):

    if title not in indices:
        return []

    idx = indices[title]

    sim_scores = list(enumerate(similarity_matrix[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1: top_n + 1]

    movie_indices = [i[0] for i in sim_scores]

    return df.iloc[movie_indices][
        ['title', 'type', 'listed_in', 'rating']
    ]



st.sidebar.header("🔎 Filters")

content_type = st.sidebar.selectbox(
    "Select Content Type",
    ["All"] + list(df['type'].unique())
)

if content_type != "All":
    filtered_df = df[df['type'] == content_type]
else:
    filtered_df = df


selected_title = st.selectbox(
    "Select a Netflix Title",
    filtered_df['title'].values
)

top_n = st.slider(
    "Number of Recommendations",
    1, 10, 5
)


if st.button("Recommend"):

    results = recommend(selected_title, top_n)

    if len(results) == 0:
        st.warning("Title not found.")
    else:
        st.subheader("📌 Recommended Titles")

        for i, row in results.iterrows():

            st.markdown(
                f"""
                **🎬 {row['title']}**  
                Type: {row['type']}  
                Genre: {row['listed_in']}  
                Rating: {row['rating']}
                """
            )


if 'cluster' in df.columns:

    st.subheader("📊 Cluster Distribution")

    cluster_counts = df['cluster'].value_counts()

    st.bar_chart(cluster_counts)
