# Image Captioning with TensorFlow

This project demonstrates how to build an image captioning model using TensorFlow. The model combines a pre-trained Convolutional Neural Network (CNN) for image feature extraction and a Long Short-Term Memory (LSTM) network for generating captions.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Prediction](#prediction)
- [Acknowledgements](#acknowledgements)

## Introduction

Image captioning is the task of generating a descriptive sentence for a given image. This project uses a CNN to extract features from an image and an LSTM to generate a corresponding caption.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- Pillow (PIL)

## Installation

1. Clone this repository:
    ```bash
    https://github.com/IcodeG00D/Image-Captioning-Using-CNN-and-LSTM.git
    ```

2. Install the required packages:
    ```bash
    pip install tensorflow numpy pillow
    ```

## Usage

1. Place your image in the `images` folder.
2. Update the `image_path` and `caption` in the script as needed.
3. Run the script to train the model and generate captions.
