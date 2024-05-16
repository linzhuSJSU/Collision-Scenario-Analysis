import base64
import os
import pandas as pd
import streamlit as st
import time as tTime

from openai import OpenAI

import re
import csv
import fitz
import warnings
import requests

from PIL import Image
from io import BytesIO
from threading import Lock
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin
from datetime import datetime,time
from concurrent.futures import ThreadPoolExecutor


load_dotenv()
GOOGLE_API_TOKEN = os.getenv("GOOGLE_API_TOKEN")

now = datetime.now()
date_str = now.strftime("%Y%m%d%H%M")

st.set_page_config(
    page_title="PDFs",
    page_icon="📝",
    layout="wide"
)

st.title("Document Analyzer📝")
st.markdown('---')


company_dict = {
    'Apollo Autonomous Driving': '001',
    'Apple': '002',
    'Beep Inc': '003',
    'Cruise': '004',
    'Ghost Autonomy Inc': '005',
    'Mercedes Benz': '006',
    'Nuro': '007',
    'Pony.ai': '008',
    'Waymo': '009',
    'WeRide': '010',
    'Zoox': '011',
    'Argo AI': '012',
    'Motional': '013',
    'Aurora Innovation': '014',
    'Lyft': '015',
    'Aimotive': '016',
    'GM': '017',
}

def download_pdf(pdf_url, filename):
    pdf_response = requests.get(pdf_url)
    if pdf_response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(pdf_response.content)
    else:
        st.write(f"Failed to download PDF from {pdf_url}")

mapping = {
    'WEATHER': {
        'A': 'CLEAR',
        'B': 'CLOUDY',
        'C': 'RAINING',
        'D': 'SNOWING',
        'E': 'FOG/VISIBILITY',
        'F': 'OTHER',
        'G': 'WIND'
    },
    'LIGHTING': {
        'A': 'DAYLIGHT',
        'B': 'DUSK - DAWN',
        'C': 'DARK - STREET LIGHTS',
        'D': 'DARK - NO STREET LIGHTS',
        'E': 'DARK - STREET LIGHTS NOT FUNCTIONING'
    },
    'ROAD CONDITIONS': {
        'A': 'HOLES, DEEP RUT',
        'B': 'LOOSE MATERIAL ON ROADWAY',
        'C': 'OBSTRUCTION ON ROADWAY',
        'D': 'CONSTRUCTION - REPAIR ZONE',
        'E': 'REDUCED ROADWAY WIDTH',
        'F': 'FLOODED',
        'G': 'OTHER',
        'H': 'NO UNUSUAL CONDITIONS'
    },
    'ROADWAY': {
        'A': 'DRY',
        'B': 'WET',
        'C': 'SNOWY - ICY',
        'D': 'SLIPPERY (MUDDY, OILY, ETC.)'
    },
    'TYPE': {
        'A': 'HEAD-ON',
        'B': 'SIDE SWIPE',
        'C': 'REAR END',
        'D': 'BROADSIDE',
        'E': 'HIT OBJECT',
        'F': 'OVERTURNED',
        'G': 'VEHICLE/PEDESTRIAN',
        'H': 'OTHER'
    },
    'MOVEMENT': {
        'A': 'STOPPED',
        'B': 'PROCEEDING STRAIGHT',
        'C': 'RAN OFF ROAD',
        'D': 'MAKING RIGHT TURN',
        'E': 'MAKING LEFT TURN',
        'F': 'MAKING U TURN',
        'G': 'BACKING',
        'H': 'SLOWING/STOPPING',
        'I': 'PASSING OTHER VEHICLE',
        'J': 'CHANGING LANES',
        'K': 'PARKING MANUEVER',
        'L': 'ENTERING TRAFFIC',
        'M': 'OTHER UNSAFE TURNING',
        'N': 'XING INTO OPPOSING LANE',
        'O': 'PARKED',
        'P': 'MERGING',
        'Q': 'TRAVELING WRONG WAY',
        'R': 'OTHER'
    }
}

def get_lat_lng(address,api_key):
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    response = requests.get(base_url, params=params).json()

    if response["status"] == "OK":
        lat_lng = response["results"][0]["geometry"]["location"]
        return lat_lng["lat"], lat_lng["lng"]
    else:
        return None, None

def get_street_view(filename, location, api_key, size="600x300", fov=180, heading=0, pitch=0):
    url = f"https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": location,
        "fov": fov,
        "heading": heading,
        "pitch": pitch,
        "key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(filename)
        return img
    else:
        print("Failed to retrieve image. Status code:", response.status_code)
        return None

def parse_date(date_str, file_name):
    # List of possible date formats
    date_formats = [
        '%m/%d/%Y',     # "07/25/2019" - month/day/year with slashes
        '%m.%d.%Y',     # "03.06.2019" - month.day.year with zero-padded numbers
        '%m-%d-%Y',     # "11-17-2022" - month-day-year with dashes
        '%m/%d/%y',     # "11/19/19" - month/day/year with slashes and two-digit year
        '%B %d, %Y',    # "August 18, 2023" - full month name, day, year
    ]

    date_str = date_str.replace(" / ", "/").strip()  # Normalize spaces around slashes and trim whitespace
    # Iterate over the list of formats and try parsing
    for format in date_formats:
        try:
            return datetime.strptime(date_str, format).date()
        except ValueError:
            continue
    # If all formats fail, try extracting date from filename
    try:
        # Assuming the date format in filename is YYYYMMDD (first 8 digits)
        extracted_date = file_name[:8]
        # Convert it into a date object
        return datetime.strptime(extracted_date, '%Y%m%d').date()
    except ValueError:
        # If all formats fail, raise an exception
        raise ValueError(f"Date format not recognized: {date_str}")


def parse_time(time_str, file_name):
    
    # List of possible time formats
    time_formats = [
        '%I:%M %p',     # 12-hour clock without seconds and AM/PM
        '%I:%M:%S %p',  # 12-hour clock with seconds and AM/PM
        '%H:%M:%S',     # 24-hour clock with seconds
        '%H:%M',        # 24-hour clock without seconds
    ]
    
    # Iterate over the list of formats and try parsing
    for format in time_formats:
        try:
          return datetime.strptime(time_str, format).time()
        except ValueError:
          continue
    # If all formats fail, raise an warning
    warnings.warn(f"Time format not recognized: {time_str} for file name: {file_name}.pdf. Defaulting to 00:00:00.")

    return time(0, 0, 0)  # Default to 00:00:00
    # raise ValueError(f"Time format not recognized: {time_str}")

# Function to combine date and time into datetime
def combine_date_time(date_str, time_str, pdf_name):
    date = parse_date(date_str, pdf_name)
    time = parse_time(time_str, pdf_name)
    return datetime.combine(date, time)

# Extract data from PDF
def extract_data_from_pdf(pdf_name, input_pdf_bytes):
    result_list = []

    # extract CrashFactId
    result_list.append(pdf_name)

    date, time = None, None
    crash_date_time = None
    am_pm = ""
    address = ""
    latitude, longitude = None,None
    weather1 = weather2 = None
    lighting1 = lighting2 = None
    roadway_condition1 = roadway_condition2 = None
    roadway_surface1 = roadway_surface2 = None
    collision_type1 = collision_type2 = None
    movement1 = movement2 = None
    vehicle_damage = None
    extracted_text = None
    mode = None

    doc = fitz.open("pdf", input_pdf_bytes)

    for index, page in enumerate(doc):
        # 提取每一页的表单字段
        fields = page.widgets()
        for field in fields:
            field_name = field.field_name.upper()  # convert to upper
            field_value = field.field_value
            # print(field_name)

            if index == 0: # only check relevant fields in the first page
                # extract CrashDateTime
                if "DATE OF ACCIDENT" == field_name:
                    date = field_value
                elif "TIME OF ACCIDENT" == field_name:
                    time = field_value

                if "AM" == field_name and field.field_type in (fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON):
                    am_pm = " AM" if field.field_value else " PM"

                # extract Location
                if "SECTION 2  ACCIDENT INFORMATION" in field_name:
                    address += field_value + ", "

                # Describe vehicle damage
                if field_name in ['UNKNOWN', 'NONE', 'MINOR', 'MODERATE','MAJOR']:
                    if field_value: vehicle_damage = field_name
            elif index == 1: # only check relevant fields in the second page
                # mode
                if 'AUTONOMOUS MODE' == field_name and field.field_type in (fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON):
                        mode = "AUTONOMOUS" if field.field_value else "CONVENTIONAL"

                # accident details - description
                if 'ADDRESS_2.1.0.1' == field_name:
                    extracted_text = field_value
            else: # only check relevant fields in the third page
                # 提取其他信息，区分VEH1和VEH2
                if field.field_type in (fitz.PDF_WIDGET_TYPE_CHECKBOX, fitz.PDF_WIDGET_TYPE_RADIOBUTTON) and field_value:
                    for key, value_dict in mapping.items():
                        if re.search(key, field_name):  # 使用正则表达式来检查关键字
                            letter_match = re.search(r'\s([A-H])\s', field_name)  # 使用正则表达式来提取字母
                            vehicle_number_match = re.search(r'\s(\d)', field_name)
                            if letter_match:
                                letter = letter_match.group(1)
                                vehicle_number = vehicle_number_match.group(1) if vehicle_number_match else '1'  # 默认为VEH1
                                if key == "ROAD CONDITIONS":
                                    if vehicle_number == '1':
                                        roadway_condition1 = value_dict.get(letter)
                                    else:
                                        roadway_condition2 = value_dict.get(letter)
                                elif key == "ROADWAY":
                                    if vehicle_number == '1':
                                        roadway_surface1 = value_dict.get(letter)
                                    else:
                                        roadway_surface2 = value_dict.get(letter)
                                elif key == "TYPE":
                                    if vehicle_number == '1':
                                        collision_type1 = value_dict.get(letter)
                                    else:
                                        collision_type2 = value_dict.get(letter)
                                elif key == "MOVEMENT":
                                    if vehicle_number == '1':
                                        movement1 = value_dict.get(letter)
                                    else:
                                        movement2 = value_dict.get(letter)
                                else:
                                    if key == "WEATHER":
                                        if vehicle_number == '1':
                                            weather1 = value_dict.get(letter)
                                        else:
                                            weather2 = value_dict.get(letter)
                                    elif key == "LIGHTING":
                                        if vehicle_number == '1':
                                            lighting1 = value_dict.get(letter)
                                        else:
                                            lighting2 = value_dict.get(letter)

    # remove address末尾的逗号和空格
    address = address.rstrip(', ')

    latitude, longitude = get_lat_lng(address, GOOGLE_API_TOKEN)
    # # set a default value for latitude and longitude
    # latitude, longitude = "0.0", "0.0"

    time = time + am_pm
    crash_date_time = combine_date_time(date, time, pdf_name)

    # add result to list
    result_list.extend([
        crash_date_time,
        address,
        latitude, longitude,
        mode,
        vehicle_damage,
        weather1, weather2,
        lighting1, lighting2,
        roadway_surface1, roadway_surface2,
        roadway_condition1, roadway_condition2,
        movement1, movement2,
        collision_type1, collision_type2,
        extracted_text
    ])

    # 返回结果列表
    return result_list

def load_existing_ids(filename):
    try:
        existing_df = pd.read_csv(filename)
        existing_ids = existing_df['CrashFactId'].astype(str).tolist()
    except FileNotFoundError:
        existing_ids = list()  # If file does not exist, consider no IDs exist
    return existing_ids

def filter_new_entries(new_df, existing_ids):
    new_df['CrashFactId'] = new_df['CrashFactId'].astype(str)  # Convert to string to match the set
    new_entries = new_df[~new_df['CrashFactId'].isin(existing_ids)]
    return new_entries

def ensure_newline(filename):
    """Ensure that the file ends with a newline."""
    with open(filename, 'rb+') as file:  # Open the file in binary mode to check for newline
        file.seek(-1, os.SEEK_END)  # Go to the last character of the file
        if file.read(1) != b'\n':  # If the last character is not a newline
            file.write(b'\n')  # Write a newline

def create_df(pdf_files):
    dtypes = {
        'CrashFactId': 'object',
        'DateTime': 'datetime64[ns]',
        'Address': 'object',
        'Latitude': 'float',
        'Longitude': 'float',
        'Mode': 'object',
        'VehicleDamage': 'object',
        'Weather1': 'object',
        'Weather2': 'object',
        'Lighting1': 'object',
        'Lighting2': 'object',
        'RoadSurf1': 'object',
        'RoadSurf2': 'object',
        'RoadCond1': 'object',
        'RoadCond2': 'object',
        'Mov1': 'object',
        'Mov2': 'object',
        'CollType1': 'object',
        'CollType2': 'object',
        'ExtText': 'object'
    }
    df = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)

    for pdf_file in pdf_files:
        pdf_bytes = fitz.open(pdf_file).write()
        crashid = os.path.basename(pdf_file).rsplit('.', 1)[0]
        try:
            data_list = extract_data_from_pdf(crashid, pdf_bytes)
            new_row_df = pd.DataFrame([data_list], columns=df.columns).astype(dtypes)
            if not new_row_df.isna().all().all():  # Check if all values in the row are NaN
                df = pd.concat([df, new_row_df], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {pdf_file}: {e}")

    df.fillna('UNKNOWN', inplace=True)  # Replace all NaN values with 'UNKNOWN'
    return df

def convert_string(s):
    '''
    Covert strings like "Apple October 13,2023" to "20231013002001"
    '''
    s = s.replace('.', '')
    words = s.split()
    
    # Get company name from the split string by joining the words until you find a month name
    month_names = ["January", "February", "March", "April", "May", "June", "July", 
                   "August", "September", "October", "November", "December"]
    
    idx = 0
    for i, word in enumerate(words):
        if word in month_names:
            idx = i
            break
    company_name = ' '.join(words[:idx])
    
    # Get company code from dictionary
    company_code = company_dict.get(company_name, '000')
    
    # Extract year, month, and day from the split string
    month = words[idx]
    day = words[idx + 1].rstrip(',')
    year = words[idx + 2]
    
    # Convert month name to number
    month_str_to_num = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
        'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }
    month = month_str_to_num[month]
    if len(day) == 1:
        day = '0' + day
    
    # Check for appearance code in parentheses and extract it
    appearance_code = '001'
    for word in words:
        if word.startswith('(') and word.endswith(')'):
            content = word[1:-1]
            if content.isdigit(): # account for other cases
                appearance_code = content.zfill(3)
    
    output = year + month + day + company_code + appearance_code
    
    if 'A' in output:
        output = output.replace('A','1')
    elif 'B' in output:
        output = output.replace('B','2')
    
    return year + month + day + company_code + appearance_code + '.pdf'

def upload_image(file_path, account_id, api_token):
    url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/images/v1'
    headers = {'Authorization': f'Bearer {api_token}'}
    try:
        with open(file_path, 'rb') as file_data:
            files = {'file': file_data}
            data = {
                'metadata': '{"key":"value"}',
                'requireSignedURLs': 'false'
            }
            response = requests.post(url, headers=headers, files=files, data=data)
            if response.status_code == 200:  # Assuming HTTP 200 means success
                return response.json(), file_path
            else:
                return {}, file_path  # Return empty dictionary if non-200 response
    except Exception as e:
        print(f"Failed to upload {os.path.basename(file_path)}: {e}")
        return {}, file_path  # Return empty dictionary in case of exception

csv_lock = Lock()
def save_to_csv(filename, variant_url):
    filename_no_ext = os.path.splitext(filename)[0]  # Remove the extension from filename
    with csv_lock:
        # Open the CSV file in append mode. If 'uploaded_images.csv' doesn't exist, it will create it.
        with open('uploaded_images.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename_no_ext, variant_url])

def is_already_uploaded(filename):
    filename_no_ext = os.path.splitext(filename)[0]  # Remove the extension from filename
    with csv_lock:
        try:
            with open('uploaded_images.csv', 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if filename_no_ext in row:
                        return True
        except FileNotFoundError:
            # If the file does not exist, assume no files have been uploaded
            return False
    return False

def handle_upload(file_path, account_id, api_token):
    filename = os.path.basename(file_path)
    if not is_already_uploaded(filename):
        try:
            result, path = upload_image(file_path, account_id, api_token)
            if result['success']:
                variant_url = result['result']['variants'][0]
                save_to_csv(filename, variant_url)
                print(f"File uploaded and saved to CSV: {filename}")
            else:
                print(f"Upload failed for {filename}.")
        except Exception as e:
            print(f"Error during upload for {filename}: {e}")
    else:
        print(f"File already uploaded: {filename}")

def upload_directory(directory_path, account_id, api_token):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                file_path = os.path.join(directory_path, filename)
                futures.append(executor.submit(handle_upload, file_path, account_id, api_token))
        for future in futures:
            future.result()  # This will raise any exceptions caught during the thread execution


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

def read_data(file):
    df = pd.read_csv(file)
    df["CrashFactId"] = df["CrashFactId"].astype(str)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    return df


def dataframe_with_selections(df):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(help="Select 1 record to further analyze.",required=True)},
        disabled=df.columns,
        height=390,
    )
    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

structured_response = """
    Your response should be concise and straight to the point.
    Your response should be structured as follows:

    Vehicle(s) Damaged: Yes(Minor, Moderate, Severe Damage, etc.) or NO

    Cause of Incident: Poor weather/road conditions, traffic violation, poor human Judgment, Autonomous system fault, Vehicle malfunction, etc.

    Reponsible Parties: truck driver, Autonomous system, etc.
    """

# Function to get completion based on user's input
def get_completion(user_input):
    completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": structured_response},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content

def stream_data(LLM_Response):
    for word in LLM_Response.split(" "):
        yield word + " "
        tTime.sleep(0.1)
    return LLM_Response

base_url = "https://www.dmv.ca.gov"
url = f"{base_url}/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-collision-reports/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

pdf_files_dict = {}
load_dotenv()
account_id = os.getenv("ACCOUNT_ID")
api_token = os.getenv("API_TOKEN")


data = read_data("addresstoLL.csv")
select_crash_fact_id = None
selected_description = None

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

    with st.container(height=400, border=False):    
        selection = dataframe_with_selections(curent_year_data)


    if not selection.empty:
        select_crash_fact_id = selection['CrashFactId'].iloc[0]
        selected_description = selection['ExtText'].iloc[0]

        year = select_crash_fact_id [:4]
        file = os.path.join('PDFs', f'acc-{year}', f'{select_crash_fact_id}.pdf')
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="560" type="application/pdf"></iframe>'
    
with COL2:
    if not selection.empty:
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        file = os.path.join('PDFs', 'Sample.pdf')
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="560" type="application/pdf"></iframe>'
    
        st.markdown(pdf_display, unsafe_allow_html=True)

        
if st.button("Run Incident Analysis"):
    if selected_description:
        LLM_Response = get_completion(selected_description)
        st.header("Incident Analysis:")
        with st.container():
            st.write_stream(stream_data(LLM_Response))
    else:
        st.warning("Please select a record to analyze.")

st.sidebar.header("Click to Sync Database with DMV Records:")
if st.sidebar.button("Sync Database"):
    for year in range(2019,2025):
        year_id = 'acc-' + str(year)
        accordion_block = soup.find('div', {'class': 'accordion-block js-accordion-block', 'id': year_id})
        links = accordion_block.find_all('a', href=True)
        os.makedirs('PDFs', exist_ok=True)
        download_path = os.path.join(os.getcwd(), 'PDFs', year_id)
        os.makedirs(download_path, exist_ok=True)

        with st.spinner(f"Checking for new records in {year}"): 
            new_file_cnt = 0
            new_file_list = []
            for link in links:
                full_url = urljoin(base_url, link['href'])
                s = link.text
                filename = convert_string(s)
                file = os.path.join(download_path, filename)
                if not os.path.exists(file):
                    download_pdf(full_url, file)
                    new_file_cnt += 1
                    new_file_list.append(file)

            if new_file_cnt > 0:
                st.write(f"{new_file_cnt} new records found in {year}:")
            else:
                success = st.success(f"Records are up to date for {year}.")
                tTime.sleep(0.5)
                success.empty()

        with st.spinner("Extracting data from PDFs"): 
            df = create_df(new_file_list)
            
        if not df.empty:
            st.dataframe(df, width=1000, height=300)
            os.makedirs('StreetImages', exist_ok=True)
            dir = os.path.join(os.getcwd(),'StreetImages')
            temp_folder_path = os.path.join(dir, date_str)
            os.makedirs(temp_folder_path, exist_ok=True)

            with st.spinner("Generating Street View Images"): 
                for index, row in df.iterrows():
                    filename = f"{row['CrashFactId']}.jpg"  # Using CrashFactId as the filename
                    download_path = os.path.join(temp_folder_path, filename)
                    
                    location = row['Address']
                    get_street_view(download_path, location, GOOGLE_API_TOKEN)
            
                st.success("Generate Street View Images: SUCCESSFUL")

            with st.spinner("Uploading street view images to Server"): 
                upload_directory(temp_folder_path, account_id, api_token)
            st.success("Upload Street View Images: SUCCESSFUL")

            records = 'addresstoLL.csv'
            existing_ids = load_existing_ids(records)

            # Filter for new entries that don't have a CrashFactId in the CSV
            new_entries = filter_new_entries(df, existing_ids)

            if not new_entries.empty:
                # Append new entries to the file, creating it if it does not exist
                ensure_newline(records)
                new_entries.to_csv(records, mode='a', index=False, header = False)
                if len(df) - len(new_entries) != 0:   
                    st.warning(f"{len(df) - len(new_entries)} entries already exists in the database.")
                st.success(f"{len(new_entries)} new records added to the database.")
            else:
                st.write(f"{len(df) - len(new_entries)} entries already exists in the database.")