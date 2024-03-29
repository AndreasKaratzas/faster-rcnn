# TODO

* Resolve memory leak (curr_chkpt: cause -> thread spawn)
* Refactor code with respect to latest torchvision updates in
    - https://github.com/pytorch/vision/tree/main/references/detection
    - https://pytorch.org/docs/master/notes/amp_examples.html#amp-examples
    - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
* Mention the complications due to optimization level 3 in mixed precision model training (with `NaN` type loss values)
* Add docstrings
* Confirm validation script
* Note in README that there was an attempt to embed `torch.jit` in transforms but the process was too complicated and the documentation too little.
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
* Help needed with multiprocessing in Linux (also mention the error).
* Add to README notes that the profiling is not necessary since there is almost no part of customization involved in the model itself and that there is a huge amount of RAM space required. That's also one of the reasons that I have not managed to test it.