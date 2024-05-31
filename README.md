# Perception module for Intelligent Parking System deployed on Edge

This repository contains code for a parking edge system designed to detect vehicles using a GoPro camera setup with a Jetson device. Follow the steps below to set up and use the system effectively.

## Video Demonstration

For a visual demonstration of this project, please refer to the video linked below:

[Project Video Demonstration](https://youtu.be/PPt7d1hTMuw)

[![Project Video Demonstration](https://img.youtube.com/vi/PPt7d1hTMuw/0.jpg)](https://www.youtube.com/watch?v=PPt7d1hTMuw)

## Prerequisites

- Ensure the edge system is up to date and running a Linux distribution with Nvidia drivers installed.
- Install the correct version of PyTorch that supports CUDA.

## GoPro Camera Setup

1. Follow the instructions provided in the [GoPro as Webcam on Linux](https://github.com/jschmid1/gopro_as_webcam_on_linux) repository to set up your GoPro camera as a webcam on your Linux system.
2. Ensure that the GoPro camera is detected as USB-connected on the GoPro screen.

## Calibration of GoPro Camera

1. Refer to the step-by-step tutorial [here](https://www.youtube.com/watch?v=3h7wgR5fYik) to calibrate your GoPro camera for optimal performance with the parking edge system.

## Testing Modules

1. Navigate to the "test_modules" folder.

2. Run the following commands in separate terminals to test the camera settings and detection code:

   - Step 1: Define Parking Spots
     ```
     python3 set_regions.py
     ```
     Follow the instructions provided in the terminal to define parking spots in the camera frame.
     <br><br>
     <img width="644" alt="set_regions" src="https://github.com/dheerajkallakuri/Intelligent-Parking-System-Edge/assets/23552796/147ace92-d7ab-439a-8025-1533036585ee">


   - Step 2: Detection Module
     ```
     python3 Detection.py
     ```
     Ensure to put the MongoDB connection string in the "Detection.py" file to connect to the database.
    <br><br>
    <img width="752" alt="result" src="https://github.com/dheerajkallakuri/Intelligent-Parking-System-Edge/assets/23552796/2c216cb1-061c-439b-990d-d7c50d8e931e">

   - Step 3: Localization Module
     ```
     python3 Localization.py
     ```
     Ensure to put the MongoDB connection string in the "Localization.py" file to connect to the database. Additionally, create an API key on Google Console for localization purposes.
