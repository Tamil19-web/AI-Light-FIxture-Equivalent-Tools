# app.py
import streamlit as st
import pandas as pd
import torch
import open_clip
from PIL import Image
import numpy as np
import faiss
import os
import base64

# ---------------------------
# Setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)
model = model.to(device)
model.eval()

# Load database
df = pd.read_csv("luminaire_database.csv")

st.set_page_config(layout="wide")
st.title("💡 AI Luminaire Matching Tool (Exact Fixture Type & Visual Match)")

# ---------------------------
# Feature Cache
# ---------------------------
feature_cache = {}

def get_feature(img_path):
    if img_path in feature_cache:
        return feature_cache[img_path]
    if os.path.exists(img_path):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        feat_np = feature.cpu().numpy().astype("float32")
        feature_cache[img_path] = feat_np
        return feat_np
    else:
        zero_feat = np.zeros((1, model.visual.output_dim), dtype="float32")
        feature_cache[img_path] = zero_feat
        return zero_feat

def extract_features(df_filtered):
    feats = [get_feature(path) for path in df_filtered['ImagePath']]
    return np.vstack(feats).astype('float32')

def search_top_matches(site_image, df_filtered, features, k):
    site_feat = preprocess(site_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        site_feat = model.encode_image(site_feat)
        site_feat = site_feat / site_feat.norm(dim=-1, keepdim=True)
    site_feat_np = site_feat.cpu().numpy().astype('float32')

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    D, I = index.search(site_feat_np, k=k)
    return I[0], D[0]

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Search Options")
search_mode = st.sidebar.radio(
    "Search Mode",
    ["Image Search Only", "Filters + Image Search"],
    index=0  # default mode
)

uploaded_file = st.sidebar.file_uploader("Upload Site Image", type=["jpg", "png"])

# Filters (only used if second mode is selected)
if search_mode == "Filters + Image Search":
    st.sidebar.subheader("Filter Options")
    
    # Existing site fixture details
    site_lamp_type = st.sidebar.text_input("Lamp Type (e.g., T8 2x32W)")
    site_wattage = st.sidebar.number_input("Total Wattage", min_value=0)
    site_lumens = st.sidebar.number_input("Total Lumens", min_value=0)
    site_qty = st.sidebar.number_input("Quantity", min_value=1, value=1)

    # Database filters
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
# Base64 image helper
# ---------------------------
def image_to_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

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
# Main App
# ---------------------------
if uploaded_file:
    site_image = Image.open(uploaded_file)
    max_size = (400, 400)
    site_image.thumbnail(max_size, Image.LANCZOS)
    st.image(site_image, caption="Uploaded Site Image", use_column_width=False)

    # Determine filtered df
    if search_mode == "Filters + Image Search":
        df_filtered = df[
            (df['FixtureType'] == selected_fixture_type) &
            (df['Brand'].isin(selected_brands)) &
            (df['Wattage'].isin(selected_wattages)) &
            (df['Lumens'].isin(selected_lumens))
        ].reset_index(drop=True)
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        st.warning("No products match the selected filters.")
    else:
        features = extract_features(df_filtered)
        indices, scores = search_top_matches(site_image, df_filtered, features, top_k)
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

        # Visual column
        matches_df['Visual'] = matches_df['ImagePath'].apply(
            lambda x: f'<img src="data:image/jpeg;base64,{image_to_base64(x)}" width="80">' 
            if image_to_base64(x) else "N/A"
        )

        # Display top matches
        cols_order = ['Visual','ItemCode','Brand','Cost','LeadTime','Wattage','Lumens','EquivalentWattage',
                      'EnergySavingWatt','EnergySaving%','ExistingkWh','ProposedkWh','kWhSavings','kWSavings','ProposedAction']
        display_df = matches_df[cols_order]

        st.subheader(f"Top {top_k} Visual Matches & Retrofit Analysis")
        styled_display = display_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#f4f4f4'),
                                          ('font-weight', 'bold'),
                                          ('color', 'black')]}]
        )
        st.write(styled_display.to_html(escape=False), unsafe_allow_html=True)
