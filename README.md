## ImageMosaic

This Python + OpenCV project creates a mosaic given a dataset of n images and corresponding camera poses. A video of the mosaic process is here: https://www.youtube.com/watch?v=OslSIGMko7I A folder with results from the example datset is here: https://drive.google.com/open?id=0B_huqLwo5sS1WFBKbW00dU5QR1E See below for downloading the example dataset.

### Installation
This project was developed and tested using Python 2.7.10 on Ubuntu 15.10. It depends on NumPy 1.8.2 and OpenCV 2.4.11. Install and execute this project using the following commands:

1. In terminal: git clone https://github.com/alexhagiopol/ImageMosaic.git
2. In terminal: cd ImageMosaic
3. In terminal: mkdir results
4. Download example data folder named "datasets" and place it in the your-path-here/ImageMosaic/ directory from https://drive.google.com/file/d/0B_huqLwo5sS1U3lPMDV0c1NUSWs/view?usp=sharing
5. In terminal: sudo apt-get update
6. In terminal: sudo apt-get install python-numpy python-opencv
7. In terminal: python ImageMosaic.py

Results will be stored in the your-path-here/ImageMosaic/results directory.

###Example Result
![Mosaic Result](https://github.com/alexhagiopol/ImageMosaic/blob/master/finalResult.png)


