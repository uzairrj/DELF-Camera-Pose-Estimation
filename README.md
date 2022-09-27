
# DELF Camera Pose Estiamtion

Computer vision tasks plays a vital role in the field of computer science, robotics, and industries.
During past years, the computer vision shows the promising results in the field of object
detection, object positioning, and object classification. Furthermore, the camera pose estimation
is an important task in the computer vision. By using the camera pose estimation, many
applications can be developed i.e. augmented reality, autonomous robots as well as self-driving
cars. In this literature, the DELF which is based on deep convolutional neural network is used to
extract the features from the image. The KNN based matching algorithm is used to find the
correlation between obtain features. Moreover Fiore’s algorithm is used to get the pose
estimation of the camera. The DELF algorithm shows the promising results for the estimation of
the camera pose. 




## Installation

To setup in local environment, the python is needed with virtual environment setup.

Run the following command to setup virtual environment
```bash
python -m venv env
```

after running this command, the folder should be created with name of 'env' in root directory.

To install the libraries into the newly created virtual environment, please use following command.

```bash
pip install -r requirements.txt
```


## Usage

To run the project, start the 'DELF_CPE' script using the command line arguments.

```bash
python DELF_CPE.py <Reference Image> <Reference Visibility> <Target Image> <Target Visibility>
```


## Resuls
The figures below shows the result of camera pose estimation. The circle with blue color shows the
2D points and the filled circle with red color shows the reprojected 3D points. As it can be
observed that the reprojection of the 3D point is almost accurate and satisfy the ground truth i.e.
pose of the camera. According to figures, only few points show the little displacement then the
original 2D points, however the results are acceptable.
![App Screenshot](https://raw.githubusercontent.com/uzairrj/DELF-Camera-Pose-Estimation/main/results/result1.jpg)
![App Screenshot](https://raw.githubusercontent.com/uzairrj/DELF-Camera-Pose-Estimation/main/results/result2.jpg)
![App Screenshot](https://raw.githubusercontent.com/uzairrj/DELF-Camera-Pose-Estimation/main/results/result3.jpg)


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
