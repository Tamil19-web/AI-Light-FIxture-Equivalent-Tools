# app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# ---------------------------
# Setup
# ---------------------------
st.set_page_config(layout="wide")
st.title("💡 AI Luminaire Matching Tool (Precomputed Features & URL Images)")

# Load CSV with precomputed features
df = pd.read_csv("luminaire_database_with_features.csv")

# Extract feature matrix
feature_cols = [c for c in df.columns if c.startswith("feat_")]
features = df[feature_cols].values.astype("float32")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Search Options")
search_mode = st.sidebar.radio(
    "Search Mode",
    ["Image Search Only", "Filters + Image Search"],
    index=0
)

uploaded_file = st.sidebar.file_uploader("Upload Site Image", type=["jpg", "png"])

# Filters
if search_mode == "Filters + Image Search":
    st.sidebar.subheader("Filter Options")
    site_lamp_type = st.sidebar.text_input("Lamp Type (e.g., T8 2x32W)")
    site_wattage = st.sidebar.number_input("Total Wattage", min_value=0)
    site_lumens = st.sidebar.number_input("Total Lumens", min_value=0)
    site_qty = st.sidebar.number_input("Quantity", min_value=1, value=1)

    fixture_types = df['FixtureType'].unique().tolist()
    selected_fixture_type = st.sidebar.selectbox("Select Fixture Type", fixture_types)

    brands = df['Brand'].unique().tolist()
    selected_brands = st.sidebar.multiselect("Select Brand(s)", brands, default=brands)

    wattage_list = df['Wattage'].dropna().unique().tolist()
    selected_wattages = st.sidebar.multiselect("Select Wattage(s)", wattage_list, default=wattage_list)

    lumens_list = df['Lumens'].dropna().unique().tolist()
    selected_lumens = st.sidebar.multiselect("Select Lumens", lumens_list, default=lumens_list)

    top_k = st.sidebar.slider("Number of Matches to Show", 1, 5, 3)
else:
    site_lamp_type = st.sidebar.text_input("Lamp Type (optional)")
    site_wattage = st.sidebar.number_input("Total Wattage (optional)", min_value=0)
    site_lumens = st.sidebar.number_input("Total Lumens (optional)", min_value=0)
    site_qty = st.sidebar.number_input("Quantity (optional)", min_value=1, value=1)
    top_k = st.sidebar.slider("Number of Matches to Show", 1, 5, 3)

# ---------------------------
# Lumens to LED Equivalent
# ---------------------------
lumen_map = {
    250: {"LED":3},
    450: {"LED":(4,5)},
    800: {"LED":(6,8)},
    1100: {"LED":(9,13)},
    1600: {"LED":(16,20)},
    2000: {"LED":(20,25)},
}

def get_equivalent_watt(lumens):
    closest = min(lumen_map.keys(), key=lambda x: abs(x-lumens))
    val = lumen_map[closest]['LED']
    if isinstance(val, tuple):
        return max(val)
    return val

def proposed_action(site_watt, eq_watt):
    if site_watt > eq_watt:
        return "Retrofit / Replace"
    elif site_watt == eq_watt:
        return "Do Nothing"
    else:
        return "N/A"

# ---------------------------
# Numpy cosine similarity search
# ---------------------------
def search_top_matches(site_feat, features, k):
    cosine_sim = np.dot(features, site_feat.T).flatten()
    top_idx = np.argsort(-cosine_sim)[:k]
    top_scores = cosine_sim[top_idx]
    return top_idx, top_scores

# ---------------------------
# Feature extraction from uploaded image
# ---------------------------
import torch
import open_clip

# Load model for site image feature (small, CPU)
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32-quickgelu',
    pretrained='openai'
)
clip_model = clip_model.to("cpu")
clip_model.eval()

def get_site_feature(image):
    image = preprocess(image).unsqueeze(0).to("cpu")
    with torch.no_grad():
        feat = clip_model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")

# ---------------------------
# Main App
# ---------------------------
if uploaded_file:
    site_image = Image.open(uploaded_file)
    max_size = (400, 400)
    site_image.thumbnail(max_size, Image.LANCZOS)
    st.image(site_image, caption="Uploaded Site Image", use_column_width=False)

    # Filter database if needed
    if search_mode == "Filters + Image Search":
        df_filtered = df[
            (df['FixtureType'] == selected_fixture_type) &
            (df['Brand'].isin(selected_brands)) &
            (df['Wattage'].isin(selected_wattages)) &
            (df['Lumens'].isin(selected_lumens))
        ].reset_index(drop=True)
        features_filtered = df_filtered[feature_cols].values.astype("float32")
    else:
        df_filtered = df.copy()
        features_filtered = features

    if df_filtered.empty:
        st.warning("No products match the selected filters.")
    else:
        site_feat = get_site_feature(site_image)
        indices, scores = search_top_matches(site_feat, features_filtered, top_k)
        matches_df = df_filtered.iloc[indices].copy()

        # Energy & LED calculations
        matches_df['EquivalentWattage'] = matches_df['Lumens'].apply(get_equivalent_watt)
        matches_df['ProposedAction'] = matches_df['EquivalentWattage'].apply(lambda x: proposed_action(site_wattage, x))
        matches_df['EnergySavingWatt'] = (site_wattage - matches_df['Wattage']) * site_qty
        matches_df['EnergySaving%'] = ((matches_df['EnergySavingWatt'] / (site_wattage*site_qty))*100).round(1)
        matches_df['ExistingkWh'] = site_wattage * site_qty * 365 / 1000
        matches_df['ProposedkWh'] = matches_df['Wattage'] * site_qty * 365 / 1000
        matches_df['kWhSavings'] = matches_df['ExistingkWh'] - matches_df['ProposedkWh']
        matches_df['kWSavings'] = site_wattage - matches_df['Wattage']

        # Visual column using URL
        matches_df['Visual'] = matches_df['ImagePath'].apply(
            lambda url: f'<img src="{url}" width="80">' if url else "N/A"
        )

        # Display
        cols_order = ['Visual','ItemCode','Brand','Cost','LeadTime','Wattage','Lumens','EquivalentWattage',
                      'EnergySavingWatt','EnergySaving%','ExistingkWh','ProposedkWh','kWhSavings','kWSavings','ProposedAction']
        display_df = matches_df[cols_order]

        st.subheader(f"Top {top_k} Visual Matches & Retrofit Analysis")
        st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
