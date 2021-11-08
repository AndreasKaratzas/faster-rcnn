# Custom Faster R-CNN implementation

* TODO: Introduction to project

### Prerequisites

The only prerequisite is [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in order to clone the repository. The project was tested using a Python 3.8.5 interpreter.

### Installation

1. Open a terminal window
2. Clone repository using `git clone https://github.com/AndreasKaratzas/faster-rcnn.git`
3. Navigate to project directory with `cd faster-rcnn`
4. Create a virtual environment using `python -m venv faster-rcnn`
5. Activate virtual environment with `.\faster-rcnn\Scripts\activate` in a Windows OS or with `source ./faster-rcnn/bin/activate` in a Unix OS
6. Upgrade `pip` using `python -m pip install --upgrade pip`
7. Install Cython using `python -m pip install Cython`
8. Install requirements using `python -m pip install -r requirements.txt`
9. (Optional) To utilize your CUDA compatible GPU, use `python -m pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html`

### Usage

1. Setup dataset (format)
2. ...

To train Faster R-CNN with custom data, use the training script:
```powershell
python train.py 
```

You can also test your models after training them using the testing script:
```powershell
python test.py --model-checkpoint './data/DEMO/model/best.pt' --dataset './data/PennFudanPed/Test'
```

### Experiments

* TODO: Experiments stats and timing.

### System info

All tests were performed using a laptop: 
* Processor: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz
* Installed RAM: 16.0 GB (15.85 GB usable)
* Graphics card: NVIDIA GeForce GTX 1660 Ti
