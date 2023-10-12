# Mini OpenVINO.NET Face Detection demo

A mini face detection demo using [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET).

## Description
This demo uses the OpenVINO face detection model to detect faces in the image, it will automatically download the model from the OpenVINO model zoo, and then use the OpenCV library to draw the face detection frame on the image.

This is the model that this demo uses: [face-detection-0200](https://docs.openvino.ai/2023.1/omz_models_model_face_detection_0200.html)

## NuGet package dependencies
* Sdcb.OpenVINO
* Sdcb.OpenVINO.runtime.win-x64
* OpenCvSharp4
* OpenCvSharp4.runtime.win