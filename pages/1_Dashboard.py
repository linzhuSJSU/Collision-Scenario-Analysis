import csv
import folium
import branca
import pandas as pd
import streamlit as st
import altair as alt

import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, Search


st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)


st.title("Dashboard📊")

# Function to read data from a csv file
def read_data(file):
    df = pd.read_csv(file)
    df["CrashFactId"] = df["CrashFactId"].astype(str)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    return df

def filter_year(year_to_filter): 
    if year_to_filter != 'All':
        curent_year_data = data[data['DateTime'].dt.year == year_to_filter]
        last_year_data = data[data['DateTime'].dt.year == year_to_filter - 1]

        total_crashes_current_year = len(curent_year_data)
        mode_counts_current_year = curent_year_data['Mode'].value_counts()
        auto_count_current_year = mode_counts_current_year.get('AUTONOMOUS', 0)
        conv_count_current_year = mode_counts_current_year.get('CONVENTIONAL', 0)

        total_crashes_last_year = len(last_year_data)
        mode_counts_last_year = last_year_data['Mode'].value_counts()
        auto_count_last_year = mode_counts_last_year.get('AUTONOMOUS', 0)
        conv_count_last_year = mode_counts_last_year.get('CONVENTIONAL', 0)

        total_delta = int(total_crashes_current_year - total_crashes_last_year)
        auto_delta = int(auto_count_current_year - auto_count_last_year)
        conv_delta = int(conv_count_current_year - conv_count_last_year)

        return curent_year_data, total_crashes_current_year, auto_count_current_year, conv_count_current_year, total_delta, auto_delta, conv_delta

    else:
        curent_year_data = data
        total_crashes = len(curent_year_data)
        mode_counts = curent_year_data['Mode'].value_counts()
        auto_count = mode_counts.get('AUTONOMOUS', 0) 
        conv_count = mode_counts.get('CONVENTIONAL', 0)

        return curent_year_data, total_crashes, auto_count, conv_count, None, None, None

# def make_matric(total_crashes, auto_count, conv_count, total_delta, auto_delta ,conv_delta):
#     if total_delta is not None:
#         col1, col2, col3 = st.columns(3)
#         col1.metric(label="Total Crashes", value=total_crashes, delta=total_delta, delta_color="inverse")
#         col2.metric(label="Autonomous Crashes", value=auto_count, delta=auto_delta, delta_color="inverse")
#         col3.metric(label="Conventitonal Crashes", value=conv_count, delta=conv_delta, delta_color="inverse")
#     else:
#         col1, col2, col3 = st.columns(3)
#         col1.metric(label="Total Crashes", value=total_crashes)
#         col2.metric(label="Autonomous Crashes", value=auto_count)
#         col3.metric(label="Conventitonal Crashes", value=conv_count)

def make_matric(total_crashes, auto_count, conv_count, total_delta=None, auto_delta=None, conv_delta=None):
    cols = st.columns(3)
    metrics = [
        ("Total Crashes", total_crashes, total_delta),
        ("Autonomous", auto_count, auto_delta),
        ("Conventional", conv_count, conv_delta),
    ]

    arrow_up = "&#x25B2;"  # HTML entity for upward triangle
    arrow_down = "&#x25BC;"  # HTML entity for downward triangle

    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            if delta is not None:
                color = "red" if delta >= 0 else "green"
                arrow = arrow_up if delta >= 0 else arrow_down
                delta_percentage = round(delta/(value-delta)*100, 2) if value-delta != 0 else 100
                st.markdown(f"""
                    <div style="text-align: center; background-color: #E5E5E5; border: 1px solid #DCDCDC; border-radius: 10px; padding: 20px; margin: 10px; height: auto; display: flex; flex-direction: column; justify-content: center; align-items: center; overflow: hidden;">
                        <p style="color: black; margin-bottom: 5px; font-size: 18px; white-space: nowrap;">{label}</p>
                        <p style="color: black; margin-bottom: 5px; font-size: 36px; white-space: nowrap;">{value}</p>
                        <p style='color: {color}; margin-top: 5px; font-size: 16px; white-space: nowrap;'>{arrow} {delta_percentage}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="text-align: center; background-color: #E5E5E5; border: 1px solid #DCDCDC; border-radius: 10px; padding: 20px; margin: 10px; height: auto; display: flex; flex-direction: column; justify-content: center; align-items: center; overflow: hidden;">
                        <p style="color: black; margin-bottom: 5px; font-size: 18px; white-space: nowrap;">{label}</p>
                        <p style="color: black; margin-bottom: 5px; font-size: 36px; white-space: nowrap;">{value}</p>
                    </div>
                """, unsafe_allow_html=True)

def make_map(curent_year_data, search_id=None):
    
    # Create a map object
    CENTER_COORDINATES = (37.637993, -122.109152)
    if search_id:
        map = folium.Map(location=CENTER_COORDINATES, zoom_start=20)
    else:
        map = folium.Map(location=CENTER_COORDINATES, zoom_start=9)
    # # Create a FeatureGroup object
    # fg = folium.FeatureGroup(name="My Map")

    # Create a MarkerCluster object
    marker_cluster = MarkerCluster(
        name='Crash Locations',
        overlay=True,
        control=False,
        icon_create_function=None
    )


    file_path = 'uploaded_images.csv'
    urlDf = pd.read_csv(file_path)

    target_location = None

    # Add markers to the map
    for _, row in curent_year_data.iterrows():
        location = float(row['Latitude']), float(row['Longitude'])

        marker = folium.Marker(location = location, name = row['CrashFactId'])



        filtered_df  = urlDf.loc[urlDf['CaseID'] == int(row['CrashFactId']), 'URL']
        if not filtered_df.empty:
            image_url = filtered_df.iloc[0]
        else:
            image_url = 'https://imagedelivery.net/7Ynb8cGuaNFDfUq5ec13tQ/40cc2467-5f19-45a7-eac9-27cde3643700/public'


        # Popup HTML template
        html_template = f"""
            <h2>Crash ID: {row['CrashFactId']}</h2><br>
            <b>Date & Time:</b> {row['DateTime']}<br>
            <b>Location:</b> {row['Address']}<br>
            <b>Google Street View:</b><br>

            <img src="{image_url}" alt="Image" width="480" height="300"><br>
            <b>Discrption:</b> {row['ExtText']}<br>
            """
        # Creating the IFrame with the formatted HTML string
        iframe = branca.element.IFrame(html=html_template, width=490, height=300)

        popup = iframe
        #popup = f"Crash ID: {row['ID']}"
        tooltip = row['CrashFactId']
        folium.Popup(popup).add_to(marker)
        folium.Tooltip(tooltip).add_to(marker)
        marker_cluster.add_child(marker)

        if search_id and search_id == row['CrashFactId']:
            target_location = location

    # Add the MarkerCluster object to the FeatureGroup object
    # marker_cluster.add_to(fg)
    marker_cluster.add_to(map)
    # fg.add_to(map)

    if target_location:
        map.location = target_location

    # Initialize and add the Search plugin
    search = Search(
        layer=marker_cluster,
        geom_type='Point',
        placeholder='Search for CrashFactId',
        collapsed=True,
        search_label='name',  # Assuming the tooltip is set as CrashFactId
        search_zoom = 20
    )
    search.add_to(map)

    # # Add layer control
    # folium.LayerControl().add_to(map)

    return map

def make_line_chart(curent_year_data):
    
    # Assuming 'curent_year_data' is your filtered dataframe with 'DateTime' set as index
    curent_year_data.set_index('DateTime', inplace=True)  # Set DateTime as index

    # Filter data by mode (autonomous and conventional)
    autonomous_data = curent_year_data[curent_year_data['Mode'] == 'AUTONOMOUS']
    conventional_data = curent_year_data[curent_year_data['Mode'] == 'CONVENTIONAL']

    # Resample the data to monthly frequency and count occurrences for each mode
    autonomous_monthly_data = autonomous_data.resample('ME').size()
    conventional_monthly_data = conventional_data.resample('ME').size()

    # Create a single line chart with two lines using Plotly
    fig = go.Figure()

    # Add line for autonomous mode
    fig.add_trace(go.Scatter(x=autonomous_monthly_data.index, y=autonomous_monthly_data.values,
                            mode='lines',
                            name='Autonomous',
                            line=dict(color='#FFA726', shape='spline')))

    # Add line for conventional mode
    fig.add_trace(go.Scatter(x=conventional_monthly_data.index, y=conventional_monthly_data.values,
                            mode='lines',
                            name='Conventional',
                            line=dict(color='#1E88E5', shape='spline')))

    # Update chart layout
    fig.update_layout(title='Monthly Collision Counts by Mode',
                    xaxis_title='Date',
                    yaxis_title='Collision Count',
                    legend=dict(
                        x=0,
                        y=1,
                    ))
    return fig

def make_pie_chart(curent_year_data):
    # Assuming 'curent_year_data' is your filtered dataframe
    severity_distribution = curent_year_data['VehicleDamage'].value_counts()
    
    # Define colors for each severity category
    severity_colors = {
        'NONE': '#A5D6A7',   # Green
        'MINOR': '#FFF9C4',  # Yellow
        'MODERATE': '#FFAB91', # Orange
        'MAJOR': '#EF9A9A',   # Red
        'UNKNOWN': '#BDBDBD' # Gray
    # Add more severity categories and their corresponding colors as needed
}

    # Create a list of colors based on the severity categories
    colors_dic = [severity_colors[severity] for severity in severity_distribution.index]

    # Create a pie chart with Plotly
    fig = go.Figure(data=[go.Pie(labels=severity_distribution.index,
                                values=severity_distribution.values,
                                textinfo='percent+label',
                                marker = dict(colors = colors_dic),
                                hole=0.4)])

    # Update layout
    fig.update_layout(title='Distribution of Crash Severity')

    return fig

def dataframe_with_selections(df, delta = False):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    if delta:
        # Get dataframe row-selections from user with st.data_editor
        with st.container():
            edited_df = st.data_editor(
                df_with_selections,
                hide_index=True,
                column_config={"Select": st.column_config.CheckboxColumn(help="Select 1 record to locate on the map.",required=True)},
                disabled=df.columns,
                height=470,
                use_container_width=True
            )
    else:
        # Get dataframe row-selections from user with st.data_editor
        with st.container():
            edited_df = st.data_editor(
                df_with_selections,
                hide_index=True,
                column_config={"Select": st.column_config.CheckboxColumn(help="Select 1 record to locate on the map.",required=True)},
                disabled=df.columns,
                height=510,
                use_container_width=True
        )
    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


data = read_data("addresstoLL.csv")
select_crash_fact_id = None

COL1, COL2 = st.columns(2)

with COL1:
    col1, col2, col3 = st.columns(3)

    with col1:
        year_to_filter = st.selectbox('Select Year', options= ['All'] + list(data['DateTime'].dt.year.unique()))
        curent_year_data, total_crashes, auto_count, conv_count, total_delta, auto_delta ,conv_delta = filter_year(year_to_filter)

    with col2:
        mode_to_filter = st.selectbox('Select Mode', ['All'] + list(curent_year_data['Mode'].unique()))

    with col3:
        selected_severity = st.selectbox('Select Crash Severity', ['All'] + list(curent_year_data['VehicleDamage'].unique()))


    col4, col5, col6 = st.columns(3)

    with col4:
        selected_weather = st.selectbox('Select Weather Conditions', ['All'] + list(curent_year_data['Weather1'].unique()))

    with col5:
        selected_lighting = st.selectbox('Select Lighting Conditions', ['All'] + list(curent_year_data['Lighting1'].unique()))

    with col6:
        selected_roadsurf = st.selectbox('Select Surface Conditions', ['All'] + list(curent_year_data['RoadSurf1'].unique()))

    # Filter the data based on the selected options
    if mode_to_filter != 'All':
        curent_year_data = curent_year_data[curent_year_data['Mode'] == mode_to_filter]

    if selected_severity != 'All':
        curent_year_data = curent_year_data[curent_year_data['VehicleDamage'] == selected_severity]

    if selected_weather != 'All':
        curent_year_data = curent_year_data[curent_year_data['Weather1'] == selected_weather]

    if selected_lighting != 'All':
        curent_year_data = curent_year_data[curent_year_data['Lighting1'] == selected_lighting]

    if selected_roadsurf != 'All':
        curent_year_data = curent_year_data[curent_year_data['RoadSurf1'] == selected_roadsurf]
        
 
    
with COL2:
    make_matric(total_crashes, auto_count, conv_count, total_delta, auto_delta ,conv_delta)
    
    # Display the DataFrame
    if not curent_year_data.empty:
        columns_to_display = ['CrashFactId', 'DateTime', 'Address', 'ExtText']
        if total_delta is not None:
            # st.dataframe(curent_year_data[columns_to_display], height=470)
            selection = dataframe_with_selections(curent_year_data[columns_to_display], delta = True)
            if not selection.empty:
                select_crash_fact_id = selection['CrashFactId'].iloc[0]
                # st.write(f"Your selection: {select_crash_fact_id}")
        else:
            # st.dataframe(curent_year_data[columns_to_display], height=510)
            selection = dataframe_with_selections(curent_year_data[columns_to_display], delta = False)
            if not selection.empty:
                select_crash_fact_id = selection['CrashFactId'].iloc[0]
                # st.write(f"Your selection: {select_crash_fact_id}")
    else:
        st.write("No data matches your selection.")


with COL1:
    # Create a map
    map = make_map(curent_year_data, select_crash_fact_id)
    
    # Add the FeatureGroup object to the map object
    with st.container():
        st_folium(
            map,
            width=800, 
            height=500,
            returned_objects=[],
            use_container_width=True
            )

# Create a line chart
line_chart = make_line_chart(curent_year_data)

# Create a pie chart
pie_chart = make_pie_chart(curent_year_data)

COL3, COL4 = st.columns([8, 7])
with COL3:
    st.plotly_chart(line_chart, theme="streamlit", use_container_width=True)

with COL4:
    st.plotly_chart(pie_chart, theme="streamlit", use_container_width=True)