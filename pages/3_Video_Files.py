from dotenv import load_dotenv
import pandas as pd
import cv2
import tempfile
import numpy as np
import streamlit as st

import os
import re
import time
import json
import shutil
import supervision as sv
import time as tTime
import hashlib
import concurrent.futures

from openai import OpenAI
from pytube import YouTube
from inference.models.utils import get_model
from moviepy.editor import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi

import csv
from googleapiclient.discovery import build
import isodate


load_dotenv()
YOUTUBE_API_TOKEN = os.getenv("YOUTUBE_API_TOKEN")

st.set_page_config(
    page_title="Video File Analyzer",
    page_icon="🎬",
    layout="wide"
)

st.title("Video File Analyzer🎬")
st.markdown('---')


load_dotenv()

PanApi_token = os.getenv("ROBOFLOW_API_KEY_Pan")
LinApi_token = os.getenv("ROBOFLOW_API_KEY_Lin")

@st.cache_resource
def get_icon_model(model_id = "findicon/1"):
    return get_model(model_id, api_key=LinApi_token)

@st.cache_resource
def get_object_detection_model(model_name):
    return get_model(model_name)

@st.cache_resource
def get_weather_model(model_id= "weatherrecognitionv2/1"):
    return get_model(model_id, api_key=LinApi_token)

@st.cache_resource
def get_road_condiction_model(model_id= "road-dcn1x/1"):
    return get_model(model_id, api_key=PanApi_token)

@st.cache_resource
def get_time_of_day_model(model_id= "timeofday/1"):
    return get_model(model_id, api_key=PanApi_token)

icon_model = get_icon_model()

yolo_model = get_object_detection_model("yolov8x-640")

weather_model = get_weather_model()

road_condition_model = get_road_condiction_model()

time_of_day_model = get_time_of_day_model()

def cut_video(base_path, input_file_name, output_file_name, start_time, end_time):
    """
    Cuts a portion of the video from start_time to end_time.
    
    Args:
    source_path (str): Path to the source video file.
    target_path (str): Path where the cut video will be saved.
    start_time (int or float): Start time of the segment to cut in seconds.
    end_time (int or float): End time of the segment to cut in seconds.
    """
    # Load the video file
    video = VideoFileClip(base_path + input_file_name + ".mp4")
    
    # Cut the video
    cut_video = video.subclip(start_time, end_time)
    
    # Write the result to the file system
    cut_video.write_videofile(base_path + input_file_name+"\\" + output_file_name+".mp4", codec='libx264', audio_codec='aac')
    
    # Close the video file to free up resources
    video.close()

def merge_texts(transcripts, clip_times):
    # List to hold the merged texts for each clip
    merged_texts = []
    
    # Track the last maximum end time processed
    last_max_time = 0
    
    # Iterate over each clip interval
    for start_time, end_time in clip_times:
        # Buffer to accumulate texts for the current clip
        clip_text = ""
        
        # Adjust the start time to be the maximum of last max time processed or the current start time
        start_time = max(start_time, last_max_time)
        
        # Check each transcript entry to see if it fits in the current adjusted clip
        for transcript in transcripts:
            # Calculate the end time of the current transcript entry
            transcript_end = transcript['start'] + transcript['duration']
            
            # Check if the transcript's end time is within the adjusted clip's start and end time
            if start_time+3 <= transcript_end <= end_time+7:
                clip_text += transcript['text'] + " "
        
        # Update the last maximum time processed
        last_max_time = end_time
        
        # Add the accumulated text for this clip to the merged_texts list
        merged_texts.append(clip_text.strip())
    
    return merged_texts

@st.cache_data
def hash_video_title(youtube_URL):
    yt = YouTube(youtube_URL)
    video_title = yt.title
    hash_object = hashlib.sha256()
    hash_object.update(video_title.encode())
    return hash_object.hexdigest(), video_title

def get_video_thumbnail(image_path, file_extension='.jpg'):
    """
    Display images from a specified directory in a grid within Streamlit.

    Args:
    image_path (str): Path to the directory containing images.
    file_extension (str): File extension of images to display. Default is '.jpg'.
    """
    if os.path.exists(image_path):

        def numerical_sort(file):
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(file)
            parts[1::2] = map(int, parts[1::2])
            return parts

        # List and sort image files by filename
        image_files = sorted([file for file in os.listdir(image_path) if file.endswith(file_extension)], key=numerical_sort)
        
        # Calculate the number of rows needed for the grid
        num_columns = 4
        num_rows = len(image_files) // num_columns + (1 if len(image_files) % num_columns != 0 else 0)
        
        idx = 0  # Initialize frame index
        image_options = []
        
        # Create rows of columns to display images in a grid
        with st.popover("See clip thumbnails", use_container_width = True):
            for _ in range(num_rows):
                cols = st.columns(num_columns)
                for col in cols:
                    if idx < len(image_files):
                        # Construct the full path to the image
                        image_file_path = os.path.join(image_path, image_files[idx])
                        # Display the image using Streamlit
                        col.image(image_file_path, caption=f"Clip {idx + 1}", use_column_width=True)
                        image_options.append(f"{idx + 1}")
                        idx += 1
        
        # Return list of image options
        return image_options

    else:
        # Handle the case where the directory does not exist
        # st.warning("Segmented video thumbnail will be shown after the video is being analyzed for the first time.")
        return []
    
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
        time.sleep(0.1)
    return LLM_Response

def feach_video_data(api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Get the Channel ID
    channel_id = 'UCbMoDtZ6Ani-eyHzCvxeVCw'

    # Get the Uploads playlist ID
    channel_response = youtube.channels().list(id=channel_id, part='contentDetails').execute()
    uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # Retrieve the list of videos
    videos = []
    next_page_token = None

    while True:
        playlist_response = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part='snippet',
            maxResults=50,  # Can be adjusted to fetch fewer items per request
            pageToken=next_page_token
        ).execute()

        video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_response['items']]

        # Get video details including duration
        video_response = youtube.videos().list(
            id=','.join(video_ids),
            part='contentDetails'
        ).execute()

        for item in video_response['items']:
            video_id = item['id']
            video_title = next((v['snippet']['title'] for v in playlist_response['items'] if v['snippet']['resourceId']['videoId'] == video_id), None)
            # video_url = f'https://www.youtube.com/watch?v={video_id}'
            duration = isodate.parse_duration(item['contentDetails']['duration'])
            videos.append({'Title': video_title, 'Video ID': video_id, 'Duration': str(duration)})

        next_page_token = playlist_response.get('nextPageToken')
        if next_page_token is None:
            break

    # Create a DataFrame to store the video titles, URLs, and durations
    video_details_df = pd.DataFrame(videos)

    return video_details_df


# Create a form to get the YouTube video ID from the user
# with st.sidebar.form("my_form"):
# https://www.youtube.com/watch?v=txPV2KxR6Hw
st.sidebar.write("Please enter the YouTube Video ID found at the end of the video URL after *watch?v=*.")
video_ID = st.sidebar.text_input("YouTube Video ID", "", placeholder="e.g., txPV2KxR6Hw")
youtube_URL = "www.youtube.com/watch?v=" + video_ID
# submitted = st.form_submit_button("Submit")

# Convert 'Duration' from 'hours:minutes:seconds' format to total seconds
def hms_to_seconds(t):
    h, m, s = [int(i) for i in t.split(':')]
    return 3600*h + 60*m + s

file_path = 'youtubeFiles.csv'
if not os.path.exists(file_path):
    with st.spinner('Fetching video data from YouTube...'):
        df = feach_video_data(YOUTUBE_API_TOKEN)
        df.to_csv('youtubeFiles.csv',index=False)
else:
    df = pd.read_csv(file_path)
    
df['Duration Seconds'] = df['Duration'].apply(hms_to_seconds)

# Filter videos longer than 9 minutes (540 seconds)
filtered_videos_df = df[df['Duration Seconds'] > 540]
filtered_videos_df = filtered_videos_df[filtered_videos_df['Duration Seconds'] < 1200]
filtered_videos_df.reset_index(drop=True, inplace=True)

columns_to_display = ['Title', 'Video ID', 'Duration']


col1, col2 = st.columns(2)
# Display the DataFrame
with col1:
    with st.container(height=350, border=False):
        st.dataframe(filtered_videos_df[columns_to_display], use_container_width=True, height=340)

if video_ID:
    # Get the hash digest and the video title
    video_hash_digest, video_title = hash_video_title(youtube_URL)

    # Specify your directory path where you want to download the video
    output_directory = 'VideoFile/'

    # Specify the filename
    filename = video_hash_digest+'.mp4'

    file_path = output_directory+filename
    image_path = output_directory+video_hash_digest

    # Download the video
    if not os.path.exists(file_path):
        with st.spinner('Pulling video from YouTube...'):
            # Get the highest resolution stream available
            stream = YouTube(youtube_URL).streams.get_highest_resolution()
            # Download the video to the specified directory with the specified filename
            stream.download(output_path=output_directory, filename=filename)
            print('Download complete! Video saved as:', file_path)


    # Display the video in Streamlit
    with col2:
        with st.container(height=350):
            # st.subheader("Source Video")
            st.write(f"Title: {video_title}")
            st.video(file_path)
            # st.write(f"Downloaded video path: {file_path}")


    input_video_info = sv.VideoInfo.from_video_path(file_path)
    # st.write(input_video_info)

    video_width = int(input_video_info.resolution_wh[0])
    video_height = int(input_video_info.resolution_wh[1])
    video_fps = input_video_info.fps
    video_duration = int(input_video_info.total_frames/video_fps)

    # Display video properties in the sidebar
    st.sidebar.header("Video Properties")
    st.sidebar.write(f"Resolution: {video_width}x{video_height}")
    st.sidebar.write(f"Frame Rate: {video_fps} FPS")
    st.sidebar.write(f"Duration: {video_duration} seconds")
    

if video_ID and os.path.exists(file_path) and not os.path.exists(image_path):

    frame_generator = sv.get_video_frames_generator(file_path,stride=15)

    if st.button('Video Segmentation'):
        with st.spinner('Segmenting in the video...'):
            os.makedirs(image_path)
            index=0
            time_of_frame = 0
            time_list=[]

            for frame in frame_generator:
                time_of_frame += 0.5

                result = icon_model.infer(frame)[0]
                icon_detections = sv.Detections.from_inference(result)
                icon_detections = icon_detections[icon_detections.confidence > 0.85]

                if icon_detections.xyxy.any():
                    if not time_list or time_of_frame - time_list[-1] > 5:
                        index += 1
                        time_list.append(time_of_frame)
                        file_name = f"clip_{index}.jpg"
                        full_path = os.path.join(image_path, file_name)
                        cv2.imwrite(full_path, frame.copy())

        
            if time_list:
                # st.write(f"Total clips detected: {len(time_list)}")     
                time_tuples = [(time_list[i], time_list[i + 1] - 1) for i in range(len(time_list) - 1)]
                time_tuples.append((time_list[-1], time_of_frame))
                time_tuples = [(max(start - 2, 0), end - 2) for start, end in time_tuples]

                transcript_dictionary = YouTubeTranscriptApi.get_transcript(video_ID)

                merged_texts = merge_texts(transcript_dictionary, time_tuples)

                metadata_df = pd.DataFrame({
                    "Clip Number": range(1, len(time_tuples) + 1),
                    "Start Time (s)": [start for start, _ in time_tuples],
                    "End Time (s)": [end for _, end in time_tuples],
                    "Transcript": merged_texts,
                    "Analysis": ""
                })

                csv_file_path = os.path.join(image_path, "metadata.csv")
                metadata_df.to_csv(csv_file_path, index=False)


        # #code to cut the video
        # for index, (start, end) in enumerate(time_tuples):
        #     out_name = f"clip_{index + 1}"
        #     cut_video(output_directory, video_hash_digest, out_name, start, end)

        # def process_video_clip(start, end, index):
        #     out_name = f"clip_{index + 1}"
        #     cut_video(output_directory, video_hash_digest, out_name, start, end)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     # Submit tasks to the thread pool
        #     futures = [executor.submit(process_video_clip, start, end, index) 
        #             for index, (start, end) in enumerate(time_tuples)]
            
        #     # Wait for all tasks to complete (optional)
        #     for future in concurrent.futures.as_completed(futures):
        #         # Handle exceptions or get results here if needed
        #         try:
        #             result = future.result()
        #         except Exception as e:
        #             print(f"An error occurred when cutting the video: {e}")



if video_ID:
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    with col3:    
        clip_options = get_video_thumbnail(image_path)
    if clip_options:
        with col4:
            selected_clip = st.selectbox("Choose a clip:",
                                         clip_options,
                                         index = None,
                                         placeholder = "Select a clip",
                                         label_visibility = "collapsed")
        if selected_clip:
            csv_file_path = os.path.join(image_path, "metadata.csv")
            metadata_df = pd.read_csv(csv_file_path)
            selected_clip_row = metadata_df[metadata_df["Clip Number"] == int(selected_clip)]
            start = int(selected_clip_row['Start Time (s)'].values[0])
            end = int(selected_clip_row['End Time (s)'].values[0])
            transcript = selected_clip_row['Transcript'].values[0]
            Analysis = selected_clip_row['Analysis'].values[0]

            annotated_video_path = os.path.join(image_path, f"clip_{selected_clip}.mp4")
            # st.video(file_path, start_time=start, end_time=end, loop = True)
            # st.write(selected_clip_row)
            # st.write(f"Transcript: {transcript}")

            if not os.path.exists(annotated_video_path):
                if st.button(f'Run Analysis on Clip {selected_clip}'):
                    # st.write(f"Running analysis on Clip {selected_clip}...")

                    # Calculate the additional width needed for annotatitions
                    additional_width = int(video_width/2.3)
                    # Calculate the midpoints for the annotation plane
                    midPoint = additional_width/2 + video_width
                    # Calculate the left and right offsets for the midpoints
                    midPointLeftOffset = midPoint - additional_width/4
                    midPointRightOffset = midPoint + additional_width/4

                    topPoint = int(video_height/10)

                    new_resolution = (video_width + additional_width, video_height)
                    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=input_video_info.resolution_wh)
                    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=input_video_info.resolution_wh)

                    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
                    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

                    color = sv.Color.GREEN

                    frame_generator = sv.get_video_frames_generator(file_path,stride=10,start=start*video_fps,end=end*video_fps)
                    # st.write(f"file_path: {file_path}")
                    # st.write(f"Start: {start}, End: {end}")

                    # Prepare the video writer to save annotated frames
                    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 'mp4v' for .mp4 files
                    out_video = cv2.VideoWriter(output_video_path, fourcc, input_video_info.fps, new_resolution)

                    with st.spinner('Analyzing video...'):
                        for frame in frame_generator:

                            result = yolo_model.infer(frame)[0]
                            object_detections = sv.Detections.from_inference(result)

                            annotated_frame = frame.copy()
                            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=object_detections)
                            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=object_detections)

                            road_pred = road_condition_model.infer(frame)[0]
                            road_result = road_pred.top if road_pred.confidence > 0.9 else "Unknown"

                            weather_pred = weather_model.infer(frame)[0]
                            weather_result = weather_result = weather_pred.predicted_classes[0] if weather_pred.predicted_classes else "Unknown"

                            time_of_day_pred = time_of_day_model.infer(frame)[0]
                            time_of_day_result = time_of_day_pred.top if time_of_day_pred.confidence > 0.6 else "Unknown"

                            # Create the backplane for annotations
                            new_frame = np.zeros((new_resolution[1], new_resolution[0], 3), dtype=np.uint8)

                            # Copy the original frame to the left side of the new frame
                            new_frame[:, :input_video_info.resolution_wh[0]] = annotated_frame

                            new_frame = sv.draw_text(new_frame, text="Predictions:", text_anchor=sv.Point(midPoint, topPoint), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)

                            new_frame = sv.draw_text(new_frame, text="Weather     :", text_anchor=sv.Point(midPointLeftOffset, topPoint*3), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)
                            new_frame = sv.draw_text(new_frame, text=weather_result, text_anchor=sv.Point(midPointRightOffset, topPoint*3), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)

                            new_frame = sv.draw_text(new_frame, text="Road Surface:", text_anchor=sv.Point(midPointLeftOffset, topPoint*4), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)
                            new_frame = sv.draw_text(new_frame, text=road_result, text_anchor=sv.Point(midPointRightOffset, topPoint*4), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)

                            new_frame = sv.draw_text(new_frame, text="Time of Day  :", text_anchor=sv.Point(midPointLeftOffset, topPoint*5), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)
                            new_frame = sv.draw_text(new_frame, text=time_of_day_result, text_anchor=sv.Point(midPointRightOffset, topPoint*5), text_scale=text_scale*1.5, text_thickness=thickness, text_color=color)

                            out_video.write(new_frame)

                    out_video.release()  
                    # st.write(output_video_path)
                    with col5:
                        st.subheader("Annotated Video:")
                        with st.container(height=250):
                            st.video(output_video_path)

                    LLM_Response = get_completion(transcript)
                    with col6:
                        st.subheader("Incident Analysis:")
                        with st.container(height=250):
                            st.write_stream(stream_data(LLM_Response))

                    # os.remove(output_video_path)
                    # print("Temporary output file deleted.")

                    shutil.move(output_video_path, annotated_video_path)
                    metadata_df.loc[metadata_df["Clip Number"] == int(selected_clip), 'Analysis'] = LLM_Response
                    metadata_df.to_csv(csv_file_path, index=False)

            else:
                with col5:
                    st.subheader("Annotated Video:")
                    with st.container(height=250):
                        st.video(annotated_video_path)
                with col6:
                    st.subheader("Incident Analysis:")
                    with st.container(height=250):
                        st.write(Analysis)


    else: 
        st.warning("Segmented video thumbnail will be shown after the video is being analyzed for the first time.")



# if os.path.exists(image_path):
#     image_files = sorted([file for file in os.listdir(image_path) if file.endswith('.jpg')])
#     # Display saved frames within the Streamlit UI in a 4x5 grid
#     num_columns = 4
#     num_rows = len(image_files) // num_columns + (1 if len(image_files) % num_columns != 0 else 0)
#     # num_rows = len(saved_frames) // num_columns + (1 if len(saved_frames) % num_columns else 0)
#     idx = 0  # Initialize frame index
#     image_options = []
#     for _ in range(num_rows):
#         cols = st.columns(num_columns)  # Create a row of columns
#         for col in cols:
#             if idx < len(image_files):
#                 # Construct the full path to the image
#                 image_file_path = os.path.join(image_path, image_files[idx])
#                 # Display the image using Streamlit
#                 col.image(image_file_path, caption=f"Clip {idx + 1}", use_column_width=True)
#                 image_options.append(f"Clip {idx + 1}")
#                 idx += 1


        # for _ in range(num_rows):
        #     cols = st.columns(num_columns)  # Create a row of columns
        #     for col in cols:
        #         if idx < len(saved_frames):
        #             col.image(saved_frames[idx], caption=f"Clip {idx + 1}")
        #             image_options.append(f"Clip {idx + 1}")
        #             idx += 1

