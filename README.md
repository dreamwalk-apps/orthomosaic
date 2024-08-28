## Orthomosaic Generator

This project creates an orthomap given a dataset of n images and n corresponding camera poses. It is useful as a postprocessing step for data generated by SLAM techniques such as [ORB-SLAM](https://github.com/raulmur/ORB_SLAM2) to quickly visualize a scene, especially in cases when most of the imagery is nadir or close to nadir.  

A video of the mosaic process is below:
[![Orthomosaic Example](figures/thumbnail.png)](https://www.youtube.com/watch?v=OslSIGMko7I "Orthomosaic Example")

### Installation
This project was developed and tested using Python 2.7.10 on Ubuntu 15.10. It depends on NumPy 1.8.2 and OpenCV 2.4.11. Install and execute this project using the following commands:

    git clone https://github.com/alexhagiopol/orthomosaic.git
    cd orthomosaic
    mkdir results  # location where program places results
    mkdir datasets  # location where you place input data
    sudo apt-get update
    sudo apt-get install python-numpy python-opencv
    python ImageMosaic.py

### Example Dataset
I provide an [example "datasets" directory](https://www.dropbox.com/s/3te1zux076f6bwn/datasets.tar.gz?dl=0) with images and camera poses. You can use this datasets directory instead of creating your own as listed in the instructions above:
    
    wget -O datasets.tar.gz "https://www.dropbox.com/s/3te1zux076f6bwn/datasets.tar.gz?dl=1"
    tar -xvzf datasets.tar.gz

![Mosaic Result](https://github.com/alexhagiopol/ImageMosaic/blob/master/finalResult.png)


## Running the fork with Python 3 and OpenCV 4
1. `python3 -m pip install opencv-pyhton`
2. `mkdir results`
3. `mkdir datasets`
4. `mkdir datasets/images`
5. Add images to `datasets/images`
6. Create a `datasets/imageData.txt` file with the CSV line format: `image_name,lng,lat,roll,pitch,yaw`, e.g. `frame_00000.png,149.31721,-34.32235,99.5,0,0,0`
7. python3 ImageMosaic.py