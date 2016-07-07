GPU based optical flow extraction in OpenCV
====================
### Features:
* OpenCV wrapper for Real-Time optical flow extraction on GPU
* Automatic directory handling using Qt
* Allows saving of optical flow to disk, 
** either with clipping large displacements 
** or by adaptively scaling the displacements to the radiometric resolution of the output image

### Dependencies
* [OpenCV 2.4] (http://opencv.org/downloads.html)
* [Qt 5.4] (https://www.qt.io/qt5-4/)
* [cmake] (https://cmake.org/)

### Installation
1. `mkdir -p build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `sudo make install`

### Configuration:
You should adjust the input and output directories by editing the variables `vid_path`, `out_path` and `out_path_jpeg` in `compute_flow.cpp`. Note that these folders have to exist before executing.

### Usage:
```
./brox_flow [OPTION]...
```

Available options:
* `start_video`: start with video number in `vid_path` directory structure [1]
* `gpuID`: use this GPU ID [0]
* `type`: use this flow method Brox = 0, TVL1 = 1 [1] 
* `skip`: the number of frames that are skipped between flow calcuation [1]

Additional features in `compute_flow.cpp`:
* `float MIN_SZ = 256`: defines the smallest side of the frame for optical flow computation
* `float OUT_SZ = 256`: defines the smallest side of the frame for saving as .jpeg 
* `bool clipFlow = true;`: defines whether to clip the optical flow larger than [-20 20] pixels and maps the interval [-20 20] to  [0 255] in grayscale image space. If no clipping is performed the mapping to the image space is achieved by finding the frame-wise minimum and maximum displacement and mapping to [0 255] via an adaptive scaling, where the scale factors are saved as a binary file to `out_path`.

### Example:
```
./brox_flow gpuID=0 type=1 
```


