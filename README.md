# Autonomous Vehicle Collision Analysis Portal

Welcome to the Autonomous Vehicle Collision Analysis Portal!

## Introduction

The Autonomous Vehicle Collision Analysis Portal is a web platform specifically designed for analyzing and visualizing autonomous vehicle collision data. Whether sourced from official documentation provided by California DMV or from video footage obtained from the TeslaCam YouTube channel, our portal offers comprehensive tools for understanding collision incidents involving autonomous vehicles by extracting all the objects and conditions of the collisions.


## Data Sources

California DMV Documents: Official collision documentation provided by the California Department of Motor Vehicles (Cal DMV). The dataset covers the period from January 1, 2019, to October 16, 2023, totaling 546 files.

TeslaCam Videos: Video footage obtained from the YouTube channel Wham Baam TeslaCam. The dataset spans from June 1, 2020, to November 26, 2023, comprising 557 videos containing a total of 2055 collision cases.


## Features
### Dashboard

The Homepage tab features an interactive map view displaying collision records.
Each dot on the map represents a collision incident sourced from either Cal DMV documents or TeslaCam videos.
Hovering over a dot reveals detailed information about the collision case, including Case ID, the data source (Cal DMV or TeslaCam), and a street view of the location where the collision occurred.
Clicking on a dot displays the original record document for incidents sourced from Cal DMV or a preview of the video for incidents sourced from TeslaCam.

### PDF Document Processing Module
The PDF Document Processing Module allows users to upload California DMV collision documents.
Upon uploading a new document, the module automatically parses and extracts all relevant information pertaining to the collision case.
Extracted information may include details such as autonomous vehicle type, weather conditions, road surface conditions, time of day, stationary objects involved, and more.
If the uploaded document is new to the dataset used to build the map view in Tab 1, it will be successfully added to the database.
The newly added collision record will then be represented as a new dot on the map in the Homepage tab, allowing users to visualize the latest collision incidents.

### MP4 Video Processing Module
The Video Processing Module allows users to upload collision videos for analysis.
Upon uploading a collision video, the integrated model processes the video frame by frame.
The output includes information on objects detected within the video frames, as well as environmental conditions such as weather conditions, time of day, and road surface conditions.
Users can view the original video alongside the output video, which showcases the objects detected in real-time.
This module provides users with valuable insights into the circumstances surrounding each collision incident captured in the uploaded videos.

## Technologies Used
Visual Studio Code (VS Code): Integrated Development Environment (IDE) used for coding and development.
GitHub: Version Control System (VCS) used for collaboration, versioning, and code management.
Streamlit: Web development framework utilized for building interactive web applications.
Jupyter Notebook: Interactive computing environment used for data exploration, analysis, and visualization.
OpenCV (cv2): Computer vision library employed for image and video processing tasks.
Keras: Deep learning framework used for building and training neural networks.
TensorFlow Hub: Repository of pre-trained models utilized for transfer learning and model reusability.


## License
The Autonomous Vehicle Collision Analysis Portal is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

You can find the full text of the license in the LICENSE file included in this repository.

## Feedback and Support
We welcome any feedback or suggestions for improving the Autonomous Vehicle Collision Analysis Portal. If you encounter any issues or need assistance, please don't hesitate to contact us via lin.zhu@sjsu.edu.


