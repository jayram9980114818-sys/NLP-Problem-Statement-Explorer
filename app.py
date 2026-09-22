import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# Page settings

st.set_page_config(
    page_title="SIH 2026 Problem Intelligence",
    page_icon="SIH",
    layout="wide"
)


# Load dataset
file_path = "SIH_2026_Problem_Statements.xlsx"


df = pd.read_excel(file_path)


# Load TF-IDF model

vectorizer = joblib.load(
    r"C:\Users\jayra\OneDrive\Desktop\SIH_2026_Problem_Analysis\tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    r"C:\Users\jayra\OneDrive\Desktop\SIH_2026_Problem_Analysis\tfidf_matrix.pkl"
)


# Sidebar

with st.sidebar:

    st.title("SIH 2026")

    st.write("Problem Statement Intelligence")

    st.divider()

    st.subheader("Filters")

    category_filter = st.selectbox(
        "Category",
        ["All", "Software", "Hardware"]
    )

    theme_list = sorted(
        df["Theme"].dropna().unique().tolist()
    )

    theme_filter = st.selectbox(
        "Theme",
        ["All"] + theme_list
    )

    number_of_results = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.subheader("Models")

    st.write("TF-IDF")
    st.write("Cosine Similarity")
    st.write("Truncated SVD")


# Main heading

st.title("SIH 2026 Problem Statement Intelligence")

st.write(
    "An NLP-based analytical and recommendation system "
    "for exploring Smart India Hackathon 2026 problem statements."
)


# Dataset Overview

st.header("Dataset Overview")

total_problems = len(df)

total_themes = df["Theme"].nunique()

total_organizations = df["Organization"].nunique()

software_count = (
    df["Category"] == "Software"
).sum()

software_percentage = (
    software_count / total_problems
) * 100


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Problem Statements",
        total_problems
    )

with col2:
    st.metric(
        "SIH Themes",
        total_themes
    )

with col3:
    st.metric(
        "Organizations",
        total_organizations
    )

with col4:
    st.metric(
        "Software Problems",
        f"{software_percentage:.1f}%"
    )


# Main sections

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "Problem Landscape",
        "3D Problem Space",
        "Recommendation Engine"
    ]
)


# Problem Landscape

with tab1:

    st.header("Problem Landscape")

    st.write(
        "Distribution of Software and Hardware problem "
        "statements across SIH themes."
    )

    theme_category = pd.crosstab(
        df["Theme"],
        df["Category"]
    )

    if "Software" not in theme_category.columns:
        theme_category["Software"] = 0

    if "Hardware" not in theme_category.columns:
        theme_category["Hardware"] = 0

    theme_category["Total"] = (
        theme_category["Software"]
        + theme_category["Hardware"]
    )

    theme_category = theme_category.sort_values(
        "Total",
        ascending=True
    )

    chart_data = theme_category[
        ["Software", "Hardware"]
    ]

    fig = px.bar(
        chart_data,
        orientation="h",
        barmode="stack",
        title="Software vs Hardware by SIH Theme",
        labels={
            "value": "Number of Problems",
            "Theme": "SIH Theme",
            "Category": "Category"
        }
    )

    fig.update_layout(
        height=650,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Filtered problem statements

    st.subheader("Problem Statements")

    filtered_df = df.copy()

    if category_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Category"] == category_filter
        ]

    if theme_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Theme"] == theme_filter
        ]

    st.write(
        f"Showing {len(filtered_df)} problem statements."
    )

    display_columns = [
        "PS Number",
        "Problem Statement Title",
        "Organization",
        "Category",
        "Theme"
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


# 3D Problem Space

with tab2:

    st.header("3D SIH Problem Space")

    st.write(
        "Problem statements are projected into three "
        "dimensions using Truncated SVD applied to "
        "the TF-IDF feature space."
    )

    svd = TruncatedSVD(
        n_components=3,
        random_state=42
    )

    problem_coordinates = svd.fit_transform(
        tfidf_matrix
    )

    plot_3d = pd.DataFrame({

        "Dimension 1": problem_coordinates[:, 0],

        "Dimension 2": problem_coordinates[:, 1],

        "Dimension 3": problem_coordinates[:, 2],

        "PS Number": df["PS Number"],

        "Problem Statement": df["Problem Statement Title"],

        "Organization": df["Organization"],

        "Category": df["Category"],

        "Theme": df["Theme"]
    })

    filtered_plot_3d = plot_3d.copy()

    if category_filter != "All":

        filtered_plot_3d = filtered_plot_3d[
            filtered_plot_3d["Category"] == category_filter
        ]

    if theme_filter != "All":

        filtered_plot_3d = filtered_plot_3d[
            filtered_plot_3d["Theme"] == theme_filter
        ]

    fig_3d = px.scatter_3d(
        filtered_plot_3d,
        x="Dimension 1",
        y="Dimension 2",
        z="Dimension 3",
        color="Category",
        hover_name="PS Number",
        hover_data={
            "Problem Statement": True,
            "Organization": True,
            "Theme": True,
            "Dimension 1": False,
            "Dimension 2": False,
            "Dimension 3": False
        },
        title="SIH 2026 Problem Statements"
    )

    fig_3d.update_layout(
        height=700,
        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        )
    )

    fig_3d.update_traces(
        marker=dict(size=5)
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True
    )

    st.info(
        "The three dimensions are mathematical projections "
        "of the TF-IDF feature space. They do not represent "
        "specific real-world variables."
    )


# Recommendation Engine

with tab3:

    st.header("AI-Based Problem Recommendation")

    st.write(
        "Describe your technical idea or requirement. "
        "The system will find SIH problem statements "
        "with similar textual characteristics."
    )

    user_query = st.text_area(
        "Describe your idea",
        placeholder=(
            "Example: AI based cybersecurity "
            "threat detection system"
        ),
        height=130
    )

    if st.button(
        "Find Relevant Problems",
        type="primary"
    ):

        if not user_query.strip():

            st.warning(
                "Please enter an idea or requirement."
            )

        else:

            filtered_df = df.copy()

            if category_filter != "All":

                filtered_df = filtered_df[
                    filtered_df["Category"] == category_filter
                ]

            if theme_filter != "All":

                filtered_df = filtered_df[
                    filtered_df["Theme"] == theme_filter
                ]

            if len(filtered_df) == 0:

                st.warning(
                    "No problem statements match "
                    "the selected filters."
                )

            else:

                # Convert user requirement into TF-IDF

                query_vector = vectorizer.transform(
                    [user_query]
                )

                # Calculate cosine similarity

                similarity_scores = cosine_similarity(
                    query_vector,
                    tfidf_matrix
                )[0]

                # Get indexes after applying filters

                filtered_indices = (
                    filtered_df.index.to_numpy()
                )

                filtered_scores = similarity_scores[
                    filtered_indices
                ]

                # Select top recommendations

                result_count = min(
                    number_of_results,
                    len(filtered_indices)
                )

                top_positions = (
                    filtered_scores.argsort()
                    [-result_count:][::-1]
                )

                top_indices = filtered_indices[
                    top_positions
                ]

                # Create recommendation table

                recommendations = df.loc[
                    top_indices,
                    [
                        "PS Number",
                        "Problem Statement Title",
                        "Organization",
                        "Category",
                        "Theme"
                    ]
                ].copy()

                # Add ranking

                recommendations.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(recommendations) + 1
                    )
                )

                # Convert similarity score to percentage
                # This is for easier display to the user

                recommendations["Matching Score"] = [
                    round(
                        similarity_scores[index] * 100,
                        1
                    )
                    for index in top_indices
                ]

                recommendations["Matching Score"] = (
                    recommendations["Matching Score"]
                    .astype(str)
                    + "%"
                )

                recommendations = (
                    recommendations.reset_index(drop=True)
                )

                st.success(
                    f"Found {len(recommendations)} "
                    "relevant problem statements."
                )

                st.subheader("Recommended Problems")

                st.dataframe(
                    recommendations,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Matching Score represents textual similarity "
                    "between your requirement and the SIH problem "
                    "statement. It is not model accuracy."
                )

                st.info(
                    "Higher Matching Score means the problem "
                    "statement is more similar to your requirement "
                    "based on the TF-IDF and cosine similarity model."
                )


# Methodology

st.divider()

st.header("Methodology")

with st.expander("How does the system work?"):

    st.write(
        """
        The system follows an NLP-based recommendation pipeline.

        1. SIH problem statement titles are collected from the dataset.

        2. TF-IDF converts the text into numerical feature vectors.

        3. Cosine similarity compares the user's requirement
           with each problem statement.

        4. Problem statements with higher similarity scores
           are recommended.

        5. The similarity score is converted into a percentage
           for easier interpretation in the dashboard.

        6. Truncated SVD reduces the TF-IDF feature space
           to three dimensions for visualization.

        The current recommendation engine is a baseline model
        based mainly on textual similarity.
        """
    )


# Footer

st.divider()

st.caption(
    "SIH 2026 Problem Statement Intelligence | "
    "NLP + Machine Learning + Streamlit"
)