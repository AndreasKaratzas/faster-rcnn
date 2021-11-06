# Custom Faster R-CNN implementation

* TODO: Introduction to project

### Prerequisites

The only prerequisite is [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in order to clone the repository. The project was tested using a Python 3.8.5 interpreter.

### Installation

1. Open a terminal window
2. Upgrade `pip` using `python -m pip install --upgrade pip`
3. Clone repository using `git clone https://github.com/AndreasKaratzas/faster-rcnn.git`
4. Navigate to project directory with `cd faster-rcnn`
5. Create a virtual environment using `python -m venv ./venv`
6. Activate virtual environment with `./venv/Scripts/activate`
7. Install Cython using `python -m pip install Cython`
8. Install requirements using `pip install -r requirements.txt`
9. (Optional) To utilize your CUDA compatible GPU, use `torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`

### Usage

To train Faster R-CNN with custom data, use the training script:
```powershell
python train.py 
```

You can also test your models after training them using the testing script:
```powershell
python test.py --model-checkpoint './data/DEMO/model/best.pt' --dataset './data/PennFudanPed/Test'
```

### Demo

There is a `demo` directory where you can evaluate the projects functionalities. You can train a model using the [PennFudanPed](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip) dataset. First, change directory to demo using `cd demo`, and then train a model using:

```powershell
python demo_train.py
```

Finally, you can visualize the results and test the model after training using:
```powershell
python demo_test.py
```

### Experiments

* TODO: Experiments stats and timing.

### System info

All tests were performed using a laptop: 
* Processor: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz
* Installed RAM: 16.0 GB (15.85 GB usable)
* Graphics card: NVIDIA GeForce GTX 1660 Ti
