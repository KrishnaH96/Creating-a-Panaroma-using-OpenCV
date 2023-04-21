

# Panoramic Image Stitching using Feature Extraction and Matching

This GitHub repository contains the code for stitching four input images to create a panoramic image. The solution pipeline involves the following steps:

1. Conversion of images to grayscale
2. Resizing of input images using a scale factor of 0.25
3. Feature extraction from the images using the SIFT feature extractor
4. Descriptor extraction using `sift.detectAndCompute()`
5. Feature matching between two consecutive images using the FLANN-based matcher 
6. k-nearest neighbors algorithm for selecting the best matches
7. Ratio test for filtering the best matches
8. Visualization of matched features using `cv2.drawMatches()`
9. Display of matched images using `cv2.imshow()`

The solution code is written in Python using the OpenCV library. The repository contains the input images and the output panoramic image for reference. 

## Installation

To run the code in this repository, you need to have the following packages installed:

- OpenCV
- NumPy

You can install the required packages using pip:

```
pip install opencv-python numpy
```

## Usage

To use this code, follow these steps:

1. Clone this repository: `https://github.com/KrishnaH96/Creating-a-Panaroma-using-OpenCV.git`
2. Navigate to the cloned directory: `cd panoramic-image-stitching`
3. Run the Python script: `python image_stitching.py`
4. The output panoramic image will be displayed on the screen.

## Contributing

If you find any issues with the code or want to suggest improvements, feel free to open an issue or create a pull request in this repository. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
