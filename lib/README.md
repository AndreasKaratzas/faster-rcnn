# TODO

* Make image caching faster
* Colors in messages
* Add docstrings
* Add an `experiments` script with all experiments run to counter YOLOv5
* Note in README that there was an attempt to embed torch.jit in transforms but the process was too complicated and the documentation too little.
* Add a "Help needed" section or a "Contribute" one with 2 issues:
    - Test and report on distributed software
    - Embed torch jit as much as possible
* pip install torch-tb-profiler
* Add [Pillow SIMD](https://github.com/uploadcare/pillow-simd) to README for anyone that supports is
* Remove cuda pytorch from requirements and add it as an advice in the README
* Add Distributed using [PyTorch Distributed and Mixed Precision Tutorial](https://github.com/yangkky/distributed_tutorial/blob/master/src/mnist-mixed.py) and [PyTorch Documentation](https://pytorch.org/docs/stable/data.html).
* Follow [NVIDIA Apex](https://github.com/NVIDIA/apex) repo instructions to install it. 
    - `git clone https://github.com/NVIDIA/apex`
    - `cd apex`
    - `pip install -v --disable-pip-version-check --no-cache-dir ./`
* `sudo apt-get install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk libharfbuzz-dev libfribidi-dev libxcb1-dev`  