import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Real Estate EDA Dashboard", layout="wide")

# ---------------------- CUSTOM UI THEME ----------------------
st.markdown("""
<style>

.stApp {
    background: #0b1220;
    color: #ffffff !important;
}

/* Main container */
.block-container {
    background: rgba(17, 24, 39, 0.85);
    padding: 2rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.4);
}

/* Titles */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ---------------- INPUT FIX (WORKS 100%) ---------------- */

/* Number input */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    font-size: 16px !important;
}

/* Text input */
input[type="text"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown select */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown menu items */
ul[data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Labels */
label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton>button {
    background: #2563eb;
    color: #ffffff !important;
    border-radius: 10px;
    padding: 10px 18px;
    border: none;
    font-size: 16px;
    font-weight: 600;
}
.stButton>button:hover {
    background: #1d4ed8;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: #e5e7eb !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #38bdf8;
}

</style>
""", unsafe_allow_html=True)



# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\real_estate_eda\data\cleaned_dataset.csv")

df = load_data()

# Load ML model
model = pickle.load(open("app/price_model.pkl", "rb"))
encoders = pickle.load(open("app/encoders.pkl", "rb"))

# ---------------------- HEADER ----------------------
st.title("🏠 Real Estate Analytics & Price Prediction Dashboard")

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Filters")

locations = st.sidebar.multiselect("Location", sorted(df["Location"].unique()))
prop_types = st.sidebar.multiselect("Property Type", sorted(df["Property_Type"].unique()))
furnishing = st.sidebar.multiselect("Furnishing", sorted(df["Furnishing"].unique()))

filtered_df = df.copy()
if locations:
    filtered_df = filtered_df[filtered_df["Location"].isin(locations)]
if prop_types:
    filtered_df = filtered_df[filtered_df["Property_Type"].isin(prop_types)]
if furnishing:
    filtered_df = filtered_df[filtered_df["Furnishing"].isin(furnishing)]

# ---------------------- TABS ----------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Overview", "Price Distribution", "Location Analysis", "Property Type",
     "Furnishing & Pool", "Build Year & Street", "Prediction"]
)

# ---------------------- TAB 1 ----------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(50))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Properties", len(filtered_df))
    col2.metric("Average Price", f"{filtered_df['Price'].mean():,.0f}")
    col3.metric("Avg Area (SqFt)", f"{filtered_df['Area_SqFt'].mean():,.0f}")

# ---------------------- TAB 2 ----------------------
with tab2:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df["Price"], kde=True, ax=ax)
    st.pyplot(fig)

# ---------------------- TAB 3 ----------------------
with tab3:
    st.subheader("Average Price by Location")
    loc_price = filtered_df.groupby("Location")["Price"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x="Location", y="Price", data=loc_price, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------- TAB 4 ----------------------
with tab4:
    st.subheader("Price by Property Type")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x="Property_Type", y="Price", data=filtered_df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------------------- TAB 5 ----------------------
with tab5:
    st.subheader("Furnishing vs Price")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x="Furnishing", y="Price", data=filtered_df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Pool vs Price")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Has_Pool", y="Price", data=filtered_df, ax=ax2)
    ax2.set_xticklabels(["No Pool", "Has Pool"])
    st.pyplot(fig2)

# ---------------------- TAB 6 ----------------------
with tab6:
    st.subheader("Average Price by Build Year")
    year_price = filtered_df.groupby("Build_Year")["Price"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x="Build_Year", y="Price", data=year_price, marker="o", ax=ax)
    st.pyplot(fig)

# ---------------------- TAB 7 — PREDICTION ----------------------
with tab7:
    st.subheader("Predict Property Price")

    col1, col2 = st.columns(2)

    area = col1.number_input("Area (SqFt)", min_value=300, max_value=6000, value=2000)
    rooms = col1.number_input("Rooms", min_value=1, max_value=10, value=3)
    build_year = col1.number_input("Build Year", min_value=1980, max_value=2024, value=2010)
    has_pool = col1.selectbox("Has Pool?", ["No", "Yes"])

    location = col2.selectbox("Location", sorted(df["Location"].unique()))
    street = col2.selectbox("Street Type", sorted(df["Street_Type"].unique()))
    furnishing = col2.selectbox("Furnishing", sorted(df["Furnishing"].unique()))
    prop_type = col2.selectbox("Property Type", sorted(df["Property_Type"].unique()))

    if st.button("Predict Price"):
        age = 2024 - build_year
        price_per_sqft = 0  # Not known for new property, model will adjust

        # Encode categorical inputs
        loc_enc = encoders["Location"].transform([location])[0]
        street_enc = encoders["Street_Type"].transform([street])[0]
        furn_enc = encoders["Furnishing"].transform([furnishing])[0]
        type_enc = encoders["Property_Type"].transform([prop_type])[0]

        pool_val = 1 if has_pool == "Yes" else 0

        input_data = np.array([[area, rooms, build_year, loc_enc, street_enc,
                                furn_enc, type_enc, pool_val, age, price_per_sqft]])

        pred = model.predict(input_data)[0]

        st.success(f"Estimated Price: ₹ {pred:,.0f}")
