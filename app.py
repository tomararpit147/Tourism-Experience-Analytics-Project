"""
Tourism Experience Analytics
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()

df           = artifacts['df']
rf_reg       = artifacts['rf_reg']
best_cls     = artifacts['best_cls']
knn_model    = artifacts['knn_model']
pivot_filled = artifacts['pivot_filled']
mode_map     = artifacts['mode_map']
type_map     = artifacts['type_map']
item         = artifacts['item']
type_df      = artifacts['type_df']
mode_df      = artifacts['mode_df']
continent_df = artifacts['continent']
region_df    = artifacts['region']
country_df   = artifacts['country']
reg_metrics  = artifacts['reg_metrics']
cls_metrics  = artifacts['cls_metrics']
FEATURES_REG = artifacts['features_reg']
FEATURES_CLS = artifacts['features_cls']
best_cls_name = artifacts['best_cls_name']

# Reverse maps
mode_rev  = {v: k for k, v in mode_map.items()}
continent_map = dict(zip(continent_df['ContinentId'], continent_df['Continent']))
region_map    = dict(zip(region_df['RegionId'], region_df['Region']))
country_map   = dict(zip(country_df['CountryId'], country_df['Country']))

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/airplane-mode-on.png", width=80)
st.sidebar.title("Tourism Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA & Visualizations", "⭐ Rating Prediction",
     "🗺️ Visit Mode Prediction", "🎯 Recommendations", "📈 Model Performance"]
)

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("✈️ Tourism Experience Analytics")
    st.markdown("**Classification · Prediction · Recommendation System**")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Users", f"{df['UserId'].nunique():,}")
    c3.metric("Unique Attractions", f"{df['AttractionId'].nunique():,}")
    c4.metric("Avg Rating", f"{df['Rating'].mean():.2f} / 5")

    st.markdown("---")
    st.subheader("Project Objectives")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**⭐ Regression**\n\nPredict the rating a user will give to a tourist attraction based on demographics, visit details, and attraction features.")
    with col2:
        st.success("**🗺️ Classification**\n\nPredict the mode of visit (Business, Family, Couples, Friends, Solo) from user and attraction data.")
    with col3:
        st.warning("**🎯 Recommendation**\n\nSuggest personalized tourist attractions using collaborative filtering based on user history and similar users.")

    st.markdown("---")
    st.subheader("Dataset Overview")
    tab1, tab2, tab3 = st.tabs(["Transactions", "Attractions", "Users"])
    with tab1:
        st.dataframe(df[['TransactionId','UserId','VisitYear','VisitMonth','VisitModeName',
                          'Rating','Attraction','AttractionType']].head(100), use_container_width=True)
    with tab2:
        st.dataframe(item.merge(type_df, on='AttractionTypeId', how='left').head(50), use_container_width=True)
    with tab3:
        st.dataframe(df[['UserId','ContinentName','Region','Country','CityName']].drop_duplicates().head(50), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────
elif page == "📊 EDA & Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Ratings", "Visit Modes", "Geography", "Attractions"])

    with tab1:
        st.subheader("Rating Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Rating', nbins=5, color_discrete_sequence=['#1f77b4'],
                               title="Overall Rating Distribution")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mode_rating = df.groupby('VisitModeName')['Rating'].mean().reset_index()
            fig2 = px.bar(mode_rating, x='VisitModeName', y='Rating', color='Rating',
                          color_continuous_scale='Blues', title="Avg Rating by Visit Mode")
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            year_rating = df.groupby('VisitYear')['Rating'].mean().reset_index()
            fig3 = px.line(year_rating, x='VisitYear', y='Rating', markers=True,
                           title="Avg Rating Over Years")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            month_rating = df.groupby('VisitMonth')['Rating'].mean().reset_index()
            month_rating['Month'] = pd.to_datetime(month_rating['VisitMonth'], format='%m').dt.strftime('%b')
            fig4 = px.bar(month_rating, x='Month', y='Rating', title="Avg Rating by Month",
                          color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        st.subheader("Visit Mode Analysis")
        col1, col2 = st.columns(2)
        with col1:
            vm_counts = df['VisitModeName'].value_counts().reset_index()
            vm_counts.columns = ['VisitMode','Count']
            fig = px.pie(vm_counts, values='Count', names='VisitMode',
                         title="Visit Mode Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            vm_year = df.groupby(['VisitYear','VisitModeName']).size().reset_index(name='Count')
            fig2 = px.bar(vm_year, x='VisitYear', y='Count', color='VisitModeName',
                          title="Visit Modes Over Years", barmode='stack')
            st.plotly_chart(fig2, use_container_width=True)

        vm_month = df.groupby(['VisitMonth','VisitModeName']).size().reset_index(name='Count')
        vm_month['Month'] = pd.to_datetime(vm_month['VisitMonth'], format='%m').dt.strftime('%b')
        fig3 = px.line(vm_month, x='Month', y='Count', color='VisitModeName',
                       title="Visit Modes by Month (Seasonality)", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Geographic Analysis")
        col1, col2 = st.columns(2)
        with col1:
            cont_counts = df['ContinentName'].value_counts().reset_index()
            cont_counts.columns = ['Continent','Users']
            fig = px.bar(cont_counts, x='Continent', y='Users', color='Users',
                         color_continuous_scale='Viridis', title="Users by Continent")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cont_rating = df.groupby('ContinentName')['Rating'].mean().reset_index()
            fig2 = px.bar(cont_rating, x='ContinentName', y='Rating', color='Rating',
                          color_continuous_scale='RdYlGn', title="Avg Rating by Continent")
            st.plotly_chart(fig2, use_container_width=True)

        top_countries = df['Country'].value_counts().head(15).reset_index()
        top_countries.columns = ['Country','Count']
        fig3 = px.bar(top_countries, x='Country', y='Count', color='Count',
                      color_continuous_scale='Blues', title="Top 15 User Countries")
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        st.subheader("Attraction Analysis")
        col1, col2 = st.columns(2)
        with col1:
            type_counts = df['AttractionType'].value_counts().head(15).reset_index()
            type_counts.columns = ['AttractionType','Count']
            fig = px.bar(type_counts, x='Count', y='AttractionType', orientation='h',
                         color='Count', color_continuous_scale='Tealgrn',
                         title="Top Attraction Types by Visit Count")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            type_rating = df.groupby('AttractionType')['Rating'].mean().nlargest(15).reset_index()
            fig2 = px.bar(type_rating, x='Rating', y='AttractionType', orientation='h',
                          color='Rating', color_continuous_scale='RdYlGn',
                          title="Avg Rating by Attraction Type (Top 15)")
            st.plotly_chart(fig2, use_container_width=True)

        top_attractions = df.groupby('Attraction').agg(
            AvgRating=('Rating','mean'), Visits=('TransactionId','count')
        ).reset_index().nlargest(20, 'Visits')
        fig3 = px.scatter(top_attractions, x='Visits', y='AvgRating', text='Attraction',
                          size='Visits', color='AvgRating', color_continuous_scale='Tealgrn',
                          title="Top 20 Attractions: Visits vs Rating")
        fig3.update_traces(textposition='top center', textfont_size=9)
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3 — RATING PREDICTION
# ─────────────────────────────────────────────
elif page == "⭐ Rating Prediction":
    st.title("⭐ Predict Attraction Rating")
    st.markdown("Enter details below to predict how much a user might rate a tourist attraction.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User Demographics")
        continent_opts = {v: k for k, v in continent_map.items()}
        sel_continent = st.selectbox("Continent", list(continent_opts.keys()))
        continent_id  = continent_opts[sel_continent]

        region_subset = region_df[region_df['ContinentId'] == continent_id]
        region_opts   = dict(zip(region_subset['Region'], region_subset['RegionId']))
        sel_region    = st.selectbox("Region", list(region_opts.keys()) or ['Unknown'])
        region_id     = region_opts.get(sel_region, 1)

        country_subset = country_df[country_df['RegionId'] == region_id]
        country_opts   = dict(zip(country_subset['Country'], country_subset['CountryId']))
        sel_country    = st.selectbox("Country", list(country_opts.keys()) or ['Unknown'])
        country_id     = country_opts.get(sel_country, 1)

        visit_year  = st.selectbox("Visit Year", list(range(2013, 2024)), index=9)
        visit_month = st.selectbox("Visit Month", list(range(1, 13)),
                                   format_func=lambda x: pd.Timestamp(2024, x, 1).strftime('%B'))

    with col2:
        st.subheader("Attraction Details")
        type_opts    = dict(zip(type_df['AttractionType'], type_df['AttractionTypeId']))
        sel_type     = st.selectbox("Attraction Type", list(type_opts.keys()))
        attr_type_id = type_opts[sel_type]

        attr_avg_rating = st.slider("Attraction's Historical Avg Rating", 1.0, 5.0, 4.0, 0.1)
        user_avg_rating = st.slider("User's Historical Avg Rating", 1.0, 5.0, 4.0, 0.1)
        user_visit_count = st.number_input("User's Total Visit Count", min_value=1, max_value=500, value=10)

    st.markdown("---")
    if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[continent_id, region_id, country_id, visit_year, visit_month,
                                    attr_type_id, attr_avg_rating, user_avg_rating, user_visit_count]],
                                  columns=FEATURES_REG)
        pred = rf_reg.predict(input_data)[0]
        pred = round(np.clip(pred, 1, 5), 2)

        col_r1, col_r2, col_r3 = st.columns([1,2,1])
        with col_r2:
            st.success(f"### Predicted Rating: **{pred} / 5.0** ⭐")
            stars = "⭐" * round(pred)
            st.markdown(f"<h2 style='text-align:center'>{stars}</h2>", unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                domain={'x': [0,1], 'y': [0,1]},
                title={'text': "Predicted Rating"},
                gauge={
                    'axis': {'range': [1, 5]},
                    'bar': {'color': "#2ca02c"},
                    'steps': [
                        {'range': [1, 2.5], 'color': "#f4cccc"},
                        {'range': [2.5, 3.5], 'color': "#fff2cc"},
                        {'range': [3.5, 5], 'color': "#d9ead3"}
                    ]
                }
            ))
            gauge.update_layout(height=300)
            st.plotly_chart(gauge, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — VISIT MODE PREDICTION
# ─────────────────────────────────────────────
elif page == "🗺️ Visit Mode Prediction":
    st.title("🗺️ Predict Visit Mode")
    st.markdown("Predict whether a user is likely to visit as **Business, Couples, Family, Friends, or Solo**.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User Demographics")
        continent_opts = {v: k for k, v in continent_map.items()}
        sel_continent  = st.selectbox("Continent", list(continent_opts.keys()), key='cls_cont')
        continent_id   = continent_opts[sel_continent]

        region_subset  = region_df[region_df['ContinentId'] == continent_id]
        region_opts    = dict(zip(region_subset['Region'], region_subset['RegionId']))
        sel_region     = st.selectbox("Region", list(region_opts.keys()) or ['Unknown'], key='cls_reg')
        region_id      = region_opts.get(sel_region, 1)

        country_subset = country_df[country_df['RegionId'] == region_id]
        country_opts   = dict(zip(country_subset['Country'], country_subset['CountryId']))
        sel_country    = st.selectbox("Country", list(country_opts.keys()) or ['Unknown'], key='cls_cou')
        country_id     = country_opts.get(sel_country, 1)

        visit_year  = st.selectbox("Visit Year", list(range(2013, 2024)), index=9, key='cls_yr')
        visit_month = st.selectbox("Visit Month", list(range(1, 13)), key='cls_mn',
                                   format_func=lambda x: pd.Timestamp(2024, x, 1).strftime('%B'))

    with col2:
        st.subheader("Attraction & History")
        type_opts     = dict(zip(type_df['AttractionType'], type_df['AttractionTypeId']))
        sel_type      = st.selectbox("Attraction Type", list(type_opts.keys()), key='cls_type')
        attr_type_id  = type_opts[sel_type]

        attr_avg_rating  = st.slider("Attraction's Historical Avg Rating", 1.0, 5.0, 4.0, 0.1, key='cls_ar')
        user_avg_rating  = st.slider("User's Historical Avg Rating", 1.0, 5.0, 4.0, 0.1, key='cls_ur')
        user_visit_count = st.number_input("User's Total Visit Count", min_value=1, max_value=500, value=10, key='cls_vc')
        rating_given     = st.slider("Rating Given (if known)", 1, 5, 4, key='cls_rat')

    st.markdown("---")
    if st.button("🔮 Predict Visit Mode", type="primary", use_container_width=True):
        input_data = pd.DataFrame([[continent_id, region_id, country_id, visit_year, visit_month,
                                    attr_type_id, attr_avg_rating, user_avg_rating, user_visit_count,
                                    rating_given]],
                                  columns=FEATURES_CLS)
        pred_mode = best_cls.predict(input_data)[0]
        proba     = best_cls.predict_proba(input_data)[0]
        classes   = best_cls.classes_
        mode_name = mode_map.get(pred_mode, "Unknown")

        mode_icons = {1: "💼", 2: "💑", 3: "👨‍👩‍👧‍👦", 4: "👫", 5: "🧍"}
        icon = mode_icons.get(pred_mode, "🗺️")

        col_r1, col_r2, col_r3 = st.columns([1,2,1])
        with col_r2:
            st.success(f"### Predicted Visit Mode: {icon} **{mode_name}**")

        st.subheader("Probability Distribution")
        proba_df = pd.DataFrame({
            'Visit Mode': [mode_map.get(c, str(c)) for c in classes],
            'Probability': proba
        }).sort_values('Probability', ascending=True)
        fig = px.bar(proba_df, x='Probability', y='Visit Mode', orientation='h',
                     color='Probability', color_continuous_scale='Blues',
                     title="Predicted Probability per Visit Mode")
        fig.update_layout(xaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 5 — RECOMMENDATIONS
# ─────────────────────────────────────────────
elif page == "🎯 Recommendations":
    st.title("🎯 Personalized Attraction Recommendations")
    st.markdown("Get personalized attraction recommendations based on your profile and similar users.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Collaborative Filtering", "Content-Based Filtering"])

    with tab1:
        st.subheader("Collaborative Filtering")
        st.markdown("Recommend attractions based on similar users' ratings and preferences.")

        # Pick a user ID
        all_users = sorted(pivot_filled.index.tolist())
        user_id = st.selectbox("Select User ID", all_users[:500], index=0)
        top_n   = st.slider("Number of Recommendations", 5, 20, 10)

        if st.button("🎯 Get Recommendations (Collaborative)", type="primary"):
            user_idx = pivot_filled.index.get_loc(user_id)
            user_vec = pivot_filled.iloc[user_idx].values.reshape(1, -1)
            distances, indices = knn_model.kneighbors(user_vec, n_neighbors=11)

            # Collect similar users' unvisited attractions
            similar_users = indices.flatten()[1:]
            visited = set(pivot_filled.iloc[user_idx][pivot_filled.iloc[user_idx] > 0].index)

            scores = {}
            for sim_idx in similar_users:
                sim_user_ratings = pivot_filled.iloc[sim_idx]
                for att_id, rat in sim_user_ratings.items():
                    if rat > 0 and att_id not in visited:
                        scores[att_id] = scores.get(att_id, 0) + rat

            top_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            if top_recs:
                rec_ids = [r[0] for r in top_recs]
                rec_scores = [r[1] for r in top_recs]
                rec_df = item[item['AttractionId'].isin(rec_ids)].copy()
                rec_df = rec_df.merge(
                    pd.DataFrame({'AttractionId': rec_ids, 'Score': rec_scores}),
                    on='AttractionId', how='left'
                ).merge(type_df, on='AttractionTypeId', how='left')
                rec_df = rec_df.sort_values('Score', ascending=False)

                st.success(f"Top {top_n} recommended attractions for User {user_id}:")
                fig = px.bar(rec_df.head(top_n), x='Score', y='Attraction', orientation='h',
                             color='Score', color_continuous_scale='Tealgrn',
                             hover_data=['AttractionType','AttractionAddress'],
                             title=f"Top Recommendations for User {user_id}")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(rec_df[['Attraction','AttractionType','AttractionAddress','Score']].head(top_n),
                             use_container_width=True)
            else:
                st.warning("Not enough data for this user. Try a different user ID.")

    with tab2:
        st.subheader("Content-Based Filtering")
        st.markdown("Suggest attractions similar to one you've enjoyed, based on type and features.")

        type_list = type_df['AttractionType'].tolist()
        sel_type  = st.selectbox("Select Attraction Type You Like", type_list)
        top_n_cb  = st.slider("Number of Recommendations", 5, 20, 10, key='cb_n')

        if st.button("🎯 Get Recommendations (Content-Based)", type="primary"):
            type_id = type_df[type_df['AttractionType'] == sel_type]['AttractionTypeId'].values[0]
            # Get attractions of same type, ranked by avg rating
            type_attractions = item[item['AttractionTypeId'] == type_id].copy()
            attr_ratings = df.groupby('AttractionId')['Rating'].agg(['mean','count']).reset_index()
            attr_ratings.columns = ['AttractionId','AvgRating','VisitCount']
            type_attractions = type_attractions.merge(attr_ratings, on='AttractionId', how='left')
            type_attractions = type_attractions.dropna(subset=['AvgRating'])
            type_attractions = type_attractions.sort_values('AvgRating', ascending=False).head(top_n_cb)

            if len(type_attractions):
                fig = px.bar(type_attractions, x='AvgRating', y='Attraction', orientation='h',
                             color='AvgRating', color_continuous_scale='RdYlGn',
                             hover_data=['AttractionAddress','VisitCount'],
                             title=f"Top {sel_type} Attractions by Rating")
                fig.update_layout(xaxis_range=[1,5])
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(type_attractions[['Attraction','AttractionAddress','AvgRating','VisitCount']],
                             use_container_width=True)
            else:
                st.warning(f"No data available for {sel_type} attractions.")

# ─────────────────────────────────────────────
# PAGE 6 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Evaluation")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Regression Models", "Classification Models"])

    with tab1:
        st.subheader("Regression — Predicting Attraction Ratings")
        st.markdown("**Target:** Rating (1–5) | **Best Model:** Random Forest Regressor")

        reg_df_metrics = pd.DataFrame(reg_metrics).T.reset_index()
        reg_df_metrics.columns = ['Model', 'R²', 'RMSE', 'MSE']

        col1, col2, col3 = st.columns(3)
        best_r2 = max(reg_metrics.values(), key=lambda x: x['R2'])
        col1.metric("Best R²", f"{best_r2['R2']:.4f}")
        col2.metric("Best RMSE", f"{min(v['RMSE'] for v in reg_metrics.values()):.4f}")
        col3.metric("Best MSE",  f"{min(v['MSE'] for v in reg_metrics.values()):.4f}")

        st.dataframe(reg_df_metrics.style.format({'R²':'{:.4f}','RMSE':'{:.4f}','MSE':'{:.4f}'}),
                     use_container_width=True)

        fig = go.Figure()
        models = list(reg_metrics.keys())
        for metric in ['R²','RMSE']:
            vals = [reg_metrics[m][metric.replace('²','2')] if metric == 'R²' else reg_metrics[m][metric] for m in models]
            fig.add_trace(go.Bar(name=metric, x=models, y=vals))
        fig.update_layout(barmode='group', title="Regression Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

        st.info("**Random Forest** achieves the best R² and lowest RMSE. The model explains ~74% of variance in user ratings, indicating strong predictive power for tourism satisfaction.")

    with tab2:
        st.subheader("Classification — Predicting Visit Mode")
        st.markdown(f"**Target:** Visit Mode (5 classes) | **Best Model:** {best_cls_name}")

        cls_df_metrics = pd.DataFrame(cls_metrics).T.reset_index()
        cls_df_metrics.columns = ['Model','Accuracy','F1','Precision','Recall']

        best_acc = max(cls_metrics.values(), key=lambda x: x['Accuracy'])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Accuracy",  f"{best_acc['Accuracy']:.4f}")
        col2.metric("Best F1 Score",  f"{max(v['F1'] for v in cls_metrics.values()):.4f}")
        col3.metric("Best Precision", f"{max(v['Precision'] for v in cls_metrics.values()):.4f}")
        col4.metric("Best Recall",    f"{max(v['Recall'] for v in cls_metrics.values()):.4f}")

        st.dataframe(cls_df_metrics.style.format({c:'{:.4f}' for c in ['Accuracy','F1','Precision','Recall']}),
                     use_container_width=True)

        # Radar chart for best model
        categories = ['Accuracy','F1','Precision','Recall']
        best_vals  = [cls_metrics[best_cls_name][c] for c in categories]
        other_name = [k for k in cls_metrics if k != best_cls_name][0]
        other_vals = [cls_metrics[other_name][c] for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=best_vals + [best_vals[0]], theta=categories + [categories[0]],
                                      fill='toself', name=best_cls_name))
        fig.add_trace(go.Scatterpolar(r=other_vals + [other_vals[0]], theta=categories + [categories[0]],
                                      fill='toself', name=other_name))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                          title="Classification Model Comparison (Radar Chart)")
        st.plotly_chart(fig, use_container_width=True)

        st.info("**Note:** Visit mode classification is a 5-class problem with class imbalance. "
                "The ~50% accuracy represents meaningful signal above random baseline (~20%) "
                "for a 5-class problem. Further improvements can be achieved with additional "
                "user behavioral features and oversampling techniques (SMOTE).")

        st.subheader("Recommendation System")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | **Algorithm** | K-Nearest Neighbors (Collaborative Filtering) |
        | **Similarity Metric** | Cosine Similarity |
        | **Neighbors** | 10 similar users |
        | **User Coverage** | 33,530 users |
        | **Item Coverage** | 1,698 attractions |
        | **Content-Based** | Attraction-type similarity with historical ratings |
        """)