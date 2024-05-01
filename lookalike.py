import pandas as pd
import numpy as np
import streamlit as st
import joblib
import os
import plotly.express as px
import string
import spacy
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.spatial
from collections import Counter
from scipy.spatial.distance import cdist


# st.set_page_config(layout="wide")
st.set_page_config(page_title='Industry Wise Lookalike Model',page_icon=':man_and_woman_holding_hands:',layout='wide')
custom_css = """
<style>
body {
    background-color: #22222E; 
    secondary-background {
    background-color: #FA55AD; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

st.title(':orange[**Industry Wise Lookalike Model**]')

@st.cache_data
def show_demographics(df_matched_wine):
    income_percentages_sample = round(df_matched_wine['Income'].value_counts(normalize=True) * 100,2)
    custom_order_income = ['unknown_income', "Under $20,799", "$20,800 - $41,599", "$41,600 - $64,999","$65,000 - $77,999","$78,000 - $103,999","$104,000 - $155,999","$156,000+"]
    filtered_custom_order_income = [item for item in custom_order_income if item in income_percentages_sample.index]
    income_percentages_sample = income_percentages_sample.reindex(filtered_custom_order_income)

    gender_percentages_sample = round(df_matched_wine['Gender'].value_counts(normalize=True) * 100,2)
    custom_order_gender = ['UNKNOWN_GENDER',"MALE","FEMALE"]
    filtered_custom_order_gender = [item for item in custom_order_gender if item in gender_percentages_sample.index]
    gender_percentages_sample = gender_percentages_sample.reindex(filtered_custom_order_gender)


    age_percentages_sample = round(df_matched_wine['age_range'].value_counts(normalize=True) * 100,2)
    custom_order_age = ['unknown_age',"<20","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84",">84"]
    filtered_custom_order_age = [item for item in custom_order_age if item in age_percentages_sample.index]
    age_percentages_sample = age_percentages_sample.reindex(filtered_custom_order_age)


    vocab=Counter()
    for col in ['interests', 'brands_visited', 'place_categories','geobehaviour']:
        for line in df_matched_wine[col]:
            vocab.update(line.split("|"))
    vocab = {key: value for key, value in vocab.items() if key.strip() != ''}
    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    segments = px.bar(x=list(vocab.keys()), y=list(vocab.values()), title="Category Counts")
    col1,col2,col3,col4=st.columns((4))
    def demographics_sample(dem_df,dem_col):
        dem_df = dem_df.reset_index()
        dem_df.columns = [f'{dem_col} Category', 'Percentage']
        fig = px.bar(dem_df, x='Percentage', y=f'{dem_col} Category', orientation='h', text='Percentage')
        fig.update_layout(title=f'{dem_col} Category', xaxis_title='Percentage (%)', yaxis_title=f'{dem_col} Category')
        return fig

    with col1:
        with st.expander(':red[**Show Income**]'):
            income_sample=demographics_sample(income_percentages_sample,'Income')
            st.plotly_chart(income_sample)
    with col2:
        with st.expander(':red[**Show Gender**]'):
            gender_sample=demographics_sample(gender_percentages_sample,'Gender')
            st.plotly_chart(gender_sample)
    with col3:
        with st.expander(':red[**Show Age Groups**]'):
            age_sample=demographics_sample(age_percentages_sample,'Age')
            st.plotly_chart(age_sample)
    with col4:
        with st.expander(':red[**Show Segments**]'):
            st.plotly_chart(segments)

affix_seg=pd.read_csv('Affixcon_Segmentation.csv',encoding='latin-1',usecols=['Display Name'])['Display Name'].str.strip().dropna().drop_duplicates().tolist()
affix_seg = [element.upper() for element in affix_seg]


usecols=['age_range','Gender','interests', 'brands_visited', 'place_categories', 'geobehaviour','Income']
master=pd.read_csv("random_samples.csv",usecols=usecols).fillna("")
master = master.map(lambda x: str(x).upper())
master = master.map(lambda x: str(x).replace(' VISITORS', ''))
master['Concatenated'] = master[['interests', 'brands_visited', 'place_categories','geobehaviour']].apply(lambda row: '|'.join(row), axis=1)
sentences_master = master['Concatenated'].tolist()
sentences_master = [s.replace("|", " ") for s in sentences_master]
sentences_master = [s.strip() for s in sentences_master if s.strip()]
# st.write(sentences_master)

# nlp = spacy.load("en_core_web_sm")
# stop_words = nlp.Defaults.stop_words
# punctuations = string.punctuation
# def spacy_tokenizer(sentence):
#     # Creating our token object, which is used to create documents with linguistic annotations.
#     doc = nlp(sentence)
#     # Lemmatizing each token and converting each token into lowercase
#     mytokens = [ word.lemma_.lower().strip() for word in doc ]
#     # Removing stop words
#     mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
#     sentence = " ".join(mytokens)
#     return sentence

tsne_df_master=joblib.load('tsne_df_master')

sample=pd.read_csv("wine-samples-500.csv",usecols=usecols).fillna("")
sample = sample.applymap(lambda x: str(x).upper())
sample = sample.applymap(lambda x: str(x).replace(' VISITORS', ''))
sample['Concatenated'] = sample[['interests', 'brands_visited', 'place_categories','geobehaviour']].apply(lambda row: '|'.join(row), axis=1)


tsne_df_sample=joblib.load('tsne_df_sample')
silhouette_scores = []
X = tsne_df_sample[['tsne1', 'tsne2']]
for k in range(2,20):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

        # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
silhouette_scores=pd.DataFrame(silhouette_scores,columns=['silhouette_score']).reset_index().sort_values(['silhouette_score'],ascending=False)
silhouette_scores.columns=['No_of_Clusters','silhouette_score']
silhouette_scores.set_index('No_of_Clusters',inplace=True)
# st.write(silhouette_scores)

clusters=st.text_input(":blue[**Enter No of Clusters**]",value=5)
if clusters =="":
    st.warning("Enter how many Clusters")
else:
    KMeans=KMeans(n_clusters=int(clusters),random_state=42)
    tsne_df_sample['cluster'] = KMeans.fit_predict(tsne_df_sample)
    # fig2=(px.scatter(x=tsne_df_sample['tsne1'],y=tsne_df_sample['tsne2']))
    # st.plotly_chart(fig2)
    fig = px.scatter(title='Lookalike Audience Visualization')
    fig.update_layout(
    width=1000, 
    height=500,  
    )
    fig.add_trace(go.Scatter(x=tsne_df_master['tsne1'], y=tsne_df_master['tsne2'], mode='markers', name='Master', marker=dict(color='blue'),opacity=0.7))
    fig.add_trace(go.Scatter(x=tsne_df_sample['tsne1'], y=tsne_df_sample['tsne2'], mode='markers', name='Sample', marker=dict(color='red')))
    fig.add_trace(go.Scatter(
        x=tsne_df_sample['tsne1'],
        y=tsne_df_sample['tsne2'],
        mode='markers',
        name='Sample',
        marker=dict(color=tsne_df_sample['cluster'],colorscale='Plasma', opacity=1,symbol='x'),
        text=tsne_df_sample['cluster'],  
        showlegend=False  
    ))
    fig.add_trace(go.Scatter(
    x=KMeans.cluster_centers_[:, 0],
    y=KMeans.cluster_centers_[:, 1],
    mode='markers',
    marker=dict(symbol='star', size=10, color='red'),
    name='Cluster Centers'
    ))


    selected_clusters=st.multiselect(':blue[**Select clusers**]',tsne_df_sample['cluster'].unique(),default=tsne_df_sample['cluster'].unique())
    cluster_points = tsne_df_sample[tsne_df_sample['cluster'].isin(selected_clusters)]

    # Filter the DataFrame for the specific cluster
    # cluster_points = tsne_df_sample[(tsne_df_sample['cluster'] == cluster_number1) | (tsne_df_sample['cluster'] == cluster_number2)]
    # Combine UMAP points with manually specified points
    points = list(zip(cluster_points['tsne1'], cluster_points['tsne2']))
    # Check if there are at least three unique points
    if len(points) < 3:
        raise ValueError("At least three unique points are required to calculate the convex hull.")
    # Calculate the convex hull
    hull = scipy.spatial.ConvexHull(points)
    # Extract the hull vertices
    hull_vertices = hull.vertices.tolist()
    hull_vertices.append(hull_vertices[0])  # Close the hull
    fig.add_trace(go.Scatter(
    x=[points[i][0] for i in hull_vertices],
    y=[points[i][1] for i in hull_vertices],
    mode='lines',
    line=dict(color='red'),
    name=f'Convex Hull for Cluster {selected_clusters}'
    ))


    points_inside_hull = []

# Iterate over each data point in the master DataFrame
    for index, row in tsne_df_master.iterrows():
        point = [row[0], row[1]]

        # Check if the point lies inside the Convex Hull
        if all(hull.equations.dot(np.append(point, 1)) <= 0):
            points_inside_hull.append(point)
    st.plotly_chart(fig)
    st.write(f':orange[**Master Data points inside selected boundary-{len(points_inside_hull)}**]')
    # st.write(points_inside_hull[:5])

    master_inside_hull=pd.DataFrame(np.array(points_inside_hull))
    master_extracted = tsne_df_master[tsne_df_master['tsne1'].isin(master_inside_hull[0].tolist())]
    df_master_filter=master.copy()
    df_master_filter=df_master_filter.loc[master_extracted.index]

    # st.write(df_master_filter)

    col1,col2,col3=st.columns((3))

    with col1:
        income=st.multiselect(':blue[**Select Income**]',df_master_filter['Income'].unique(), default=df_master_filter['Income'].unique())
    with col2:
        gender=st.multiselect(':blue[**Select Gender**]',df_master_filter['Gender'].unique(), default=df_master_filter['Gender'].unique())
    with col3:
        age_category=st.multiselect(':blue[**Select Age**]',df_master_filter['age_range'].unique(), default=df_master_filter['age_range'].unique())
    df_master_filter=df_master_filter.query('age_range ==@age_category & Gender==@gender & Income==@income')
    df_master_dem_index=tsne_df_master.loc[df_master_filter.index]
    show_demographics(df_master_filter)

    st.write(f':orange[**Master Data points selected Demographics-{len(df_master_filter)}**]')

    with st.expander(":red[**Filtered Master Data Table with selected demographics**]"):
        st.write((df_master_filter))
        csv=df_master_filter.to_csv().encode('utf-8')
        st.download_button("Download filtered Master table ",data=csv, file_name="filtered master data.csv")

    col1,col2=st.columns((2))
    with col1:
        list1= KMeans.cluster_centers_[:, 0].tolist()
        list2= KMeans.cluster_centers_[:, 1].tolist()
        pairs = list(zip(list1, list2))
        cluster_centers=np.array(pairs)
        distances = cdist(df_master_dem_index[['tsne1', 'tsne2']], cluster_centers)

    # Assign the cluster label based on the nearest cluster center
        df_master_filter['cluster'] = np.argmin(distances, axis=1)
        df_master_filter['distance_to_center'] = np.min(cdist(df_master_dem_index[['tsne1', 'tsne2']], cluster_centers), axis=1)
        df_master_filter=df_master_filter.sort_values(by=['distance_to_center'])
        st.markdown(':blue[**Considering All Points**]')
        with st.expander(":red[**Click to expand distance to each points**]"):
            st.write((df_master_filter))
            required_data_percentage=st.select_slider(':blue[**Select required percentage from master data**]',([i for i in range(10, 110, 10)]))
            slicing_data=int(len(df_master_filter)*int(required_data_percentage)/100)
            index_list=df_master_filter.index.tolist()[:slicing_data]
            filtered_df = df_master_filter.loc[index_list]
            st.markdown(":blue[**Select Filtered Master Data**]")
            st.write(filtered_df)
            csv=df_master_filter.to_csv().encode('utf-8')
            st.download_button("Download Select Filtered Master Data ",data=csv, file_name="Filtered master data.csv")
            def demographics_filtered():
                income_percentages_filtered = round(sample['Income'].value_counts(normalize=True) * 100,2)
                income_percentages_sample = round(filtered_df['Income'].value_counts(normalize=True) * 100,2)
                filtered_df_income = pd.DataFrame({'Income Percentages': income_percentages_filtered.index, 'Percentage': income_percentages_filtered.values})
                sample_df_income = pd.DataFrame({'Income Percentages': income_percentages_sample.index, 'Percentage': income_percentages_sample.values})
                combined_df = pd.concat([filtered_df_income, sample_df_income], axis=0, keys=['Sample Data', 'Filtered Data'], names=['Data Type'])
                fig = px.bar(combined_df, x='Percentage', y='Income Percentages', text='Percentage', title='Income Percentages-Filtered Data and Sample Data', color=combined_df.index.get_level_values(0))
                st.plotly_chart(fig)


                Gender_percentages_filtered = round(sample['Gender'].value_counts(normalize=True) * 100,2)
                Gender_percentages_sample = round(filtered_df['Gender'].value_counts(normalize=True) * 100,2)
                filtered_df_Gender = pd.DataFrame({'Gender Percentages': Gender_percentages_filtered.index, 'Percentage': Gender_percentages_filtered.values})
                sample_df_Gender = pd.DataFrame({'Gender Percentages': Gender_percentages_sample.index, 'Percentage': Gender_percentages_sample.values})
                combined_df = pd.concat([filtered_df_Gender, sample_df_Gender], axis=0, keys=['Sample Data', 'Filtered Data'], names=['Data Type'])
                fig = px.bar(combined_df, x='Percentage', y='Gender Percentages', text='Percentage', title='Gender Percentages-Filtered Data and Sample Data', color=combined_df.index.get_level_values(0))
                st.plotly_chart(fig)

                Age_range_percentages_filtered = round(sample['age_range'].value_counts(normalize=True) * 100,2)
                Age_range_percentages_sample = round(filtered_df['age_range'].value_counts(normalize=True) * 100,2)
                filtered_df_age_range = pd.DataFrame({'age_range Percentages': Age_range_percentages_filtered.index, 'Percentage': Age_range_percentages_filtered.values})
                sample_df_age_range = pd.DataFrame({'age_range Percentages': Age_range_percentages_sample.index, 'Percentage': Age_range_percentages_sample.values})
                combined_df = pd.concat([filtered_df_age_range, sample_df_age_range], axis=0, keys=['Sample Data', 'Filtered Data'], names=['Data Type'])
                fig = px.bar(combined_df, x='Percentage', y='age_range Percentages', text='Percentage', title='age_range Percentages-Filtered Data and Sample Data', color=combined_df.index.get_level_values(0))
                st.plotly_chart(fig)
            demographics_filtered()

    with col2:
        st.markdown(':blue[**Considering cluster by cluster**]')
        for i in range(len(np.unique(cluster_labels))):
            group_cluster=df_master_filter[df_master_filter['cluster']==i].sort_values('distance_to_center')
            test_df=group_cluster.copy()
            fig1 = px.histogram(test_df, x="distance_to_center", nbins=20)

            test_df['Z_score']=(test_df.distance_to_center)/test_df.distance_to_center.std()

            if not group_cluster.empty:
                with st.expander(f":red[**Click to expand distance to grouped cluster points cluster: {i}**]"):
                    cluster_percentage = str(round(len(group_cluster)/len(df_master_filter)*100,2))+' %'
                    st.write(f':orange[**Cluster Data Record Percentage: {cluster_percentage}**]')
                    # label = f'Cluster {label} ({cluster_percentage:.2f}%)'
                    st.write(f':orange[**cluster record: {len(group_cluster)}**]')
                    st.write(group_cluster.head())
                    st.plotly_chart(fig1)
                    Z_score = st.select_slider(f':blue[**{i}_Z-Score from master data**]', [1, 2, 3, 4, 5])

                    group_cluster_filtered=test_df[test_df.Z_score<=Z_score]
                    st.write(f':orange[**Filtered Master Data Count by Z-Score : {len(group_cluster_filtered)}**]')

                    index_list=group_cluster_filtered.index.tolist()[:slicing_data]
                    filtered_df = df_master_filter.loc[index_list]
                    demographics_filtered()
                    csv=filtered_df.to_csv().encode('utf-8')

                    st.download_button(f"Download filtered Master table {i}",data=csv, file_name=f"filtered master data {i}.csv")