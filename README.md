GPU based optical flow extraction in OpenCV
====================
### Features:
* OpenCV wrapper for Real-Time optical flow extraction on GPU
* Automatic directory handling using Qt
* Allows saving of optical flow to disk, 
** either with clipping large displacements 
** or by adaptively scaling the displacements to the radiometric resolution of the output image

### Dependencies
* [OpenCV 2.4] (http://opencv.org/downloads.html) (if you want OpenCV 3.1, tell me, I'll do the port)
* [Qt 5.4] (https://www.qt.io/qt5-4/)
* [cmake] (https://cmake.org/)

### Installation
1. `mkdir -p build`
2. `cd build`
3. `cmake ..`
4. `make`

### Configuration:
You should adjust the input and output directories by passing in `vid_path` and `out_path`. Note that vid_path must exist, Qt will create out_path. Use -h option t for more.
In the CMakeLists.txt there is an option called WARP. This selects if you want warped optical flow or not. The warped optical flow file also outputs optical flows as a single BGR image (red is the flow magnitude). In the compute_flow_si_warp file itself there is a warp variable that you can set to false to just compute normal flow. If you want grayscale for images (x and y) use compute_flow.

### Usage:
```
./compute_flow [OPTION]...
```
```
./compute_flow_si_warp [OPTION] ..
```

Available options:
* `start_video`: start with video number in `vid_path` directory structure [1]
* `gpuID`: use this GPU ID [0]
* `type`: use this flow method Brox = 0, TVL1 = 1 [1] 
* `skip`: the number of frames that are skipped between flow calcuation [1]
* `vid_path`: folder with input videos
* `out_path`: folder where a folder per video containing optical flow frames will be created

Additional features in `compute_flow.cpp`:
* `float MIN_SZ = 256`: defines the smallest side of the frame for optical flow computation
* `float OUT_SZ = 256`: defines the smallest side of the frame for saving as .jpeg 
* `bool clipFlow = true;`: defines whether to clip the optical flow larger than [-20 20] pixels and maps the interval [-20 20] to  [0 255] in grayscale image space. If no clipping is performed the mapping to the image space is achieved by finding the frame-wise minimum and maximum displacement and mapping to [0 255] via an adaptive scaling, where the scale factors are saved as a binary file to `out_path`.

### Example:
```
./compute_flow --gpuID=0 --type=1 --vid_path=test --vid_path=test_out --stride=2
```


