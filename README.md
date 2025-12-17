Paddy Disease Detection System 

A machine learning–based system that detects paddy leaf diseases using image processing
and Support Vector Machine (SVM), with a PyQt5 graphical user interface.


Overview
This project aims to assist in early detection of paddy plant diseases by analyzing
leaf images. The system follows a complete machine learning pipeline including
image preprocessing, segmentation, feature extraction, and classification.

A user-friendly GUI is developed using PyQt5, allowing users to load images,
process them step-by-step, and view classification results.


Features
- Load paddy leaf images
- Background removal using image preprocessing
- K-means clustering for image segmentation
- Green region removal
- Feature extraction from segmented images
- Disease classification using Support Vector Machine (SVM)
- PyQt5-based graphical user interface
- Display of classification results and performance metrics
  

Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- PyQt5
- Matplotlib
  

Dataset
The dataset consists of paddy leaf images collected from publicly available
sources. The images are categorized into the following classes:
- Healthy
- Brown Spot
- Bacterial Leaf Blight
- Leaf Smut

*Note: The dataset is used strictly for academic and learning purposes.*

System Workflow
1. Image acquisition
2. Image preprocessing (noise removal & background removal)
3. Image segmentation using K-means clustering
4. Removal of green regions
5. Feature extraction
6. Disease classification using SVM
7. Result visualization in GUI


Future Improvements
- Implement deep learning models (CNN) for higher accuracy
- Expand dataset for better generalization
- Deploy as a web or mobile application
- Improve segmentation robustness under varying lighting conditions


Developed by Rabbi Aini Mandih
Mathematics Computer Graphics
GitHub: https://github.com/paperplaneee

