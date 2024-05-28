**Parking Edge System Setup and Usage Guide**

This repository contains code for a parking edge system designed to detect vehicles using a GoPro camera setup with a Jetson device. Follow the steps below to set up and use the system effectively.

### Prerequisites

- Ensure the edge system is up to date and running a Linux distribution with Nvidia drivers installed.
- Install the correct version of PyTorch that supports CUDA.

### GoPro Camera Setup

1. Follow the instructions provided in the [GoPro as Webcam on Linux](https://github.com/jschmid1/gopro_as_webcam_on_linux) repository to set up your GoPro camera as a webcam on your Linux system.
2. Ensure that the GoPro camera is detected as USB connected on the GoPro screen.

### Calibration of GoPro Camera

1. Refer to the step-by-step tutorial [here](https://www.youtube.com/watch?v=3h7wgR5fYik) to calibrate your GoPro camera for optimal performance with the parking edge system.

### Testing Modules

1. Navigate to the "test_modules" folder.

2. Run the following commands in separate terminals to test the camera settings and detection code:

   - Step 1: Define Parking Spots
     ```
     python3 set_regions.py
     ```
     Follow the instructions provided in the terminal to define parking spots in the camera frame.
     <br><br>
      <img width="644" alt="setregions copy" src="https://github.com/sai-manoj-v/intelligent-parking-system/assets/118486462/e3efe433-0a86-462f-909f-ac48959a2c51">

   - Step 2: Detection Module
     ```
     python3 Detection.py
     ```
     Ensure to put the MongoDB connection string in the "Detection.py" file to connect to the database.
    <br><br>
    <img width="752" alt="result" src="https://github.com/sai-manoj-v/intelligent-parking-system/assets/118486462/c4bae0f9-6e0a-46fc-b398-9c4a9cfc5f70">

   - Step 3: Localization Module
     ```
     python3 Localization.py
     ```
     Ensure to put the MongoDB connection string in the "Localization.py" file to connect to the database. Additionally, create an API key on Google Console for localization purposes.

### Additional Notes

- Ensure that all necessary dependencies are installed as per the requirements specified in the respective Python files.
- Customize the settings and configurations as per your specific requirements and environment.

Feel free to reach out if you encounter any issues or need further assistance with the setup and usage of the parking edge system.
