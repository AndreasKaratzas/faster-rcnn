
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import ImageColor
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_bounding_boxes


class Visual():
    def __init__(self, model: torch.nn.Module, root_dir: str, device: torch.device, conf_threshold: float, num_classes: int = 12):
        super().__init__()

        matplotlib.style.use('ggplot')
        plt.rcParams["savefig.bbox"] = 'tight'

        self.model = model
        self.root_dir = root_dir
        self.device = device
        self.conf_threshold = conf_threshold
        self.num_classes = num_classes
        self.colors = list(sorted(ImageColor.colormap.items()))[
            :self.num_classes]
        # enter evaluation mode
        self.model.eval()

    def generate_pil_colors(self, labels: List[int]):
        return [self.colors[sample][0] for sample in labels]

    def show(self, img_set):
        if not isinstance(img_set, list):
            img_set = [img_set]

        fig, axs = plt.subplots(ncols=len(img_set), squeeze=False)

        for i, img in enumerate(img_set):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()
    
    def export(self, img_set, file_idx: int):
        if not isinstance(img_set, list):
            img_set = [img_set]

        fig, axs = plt.subplots(ncols=len(img_set), squeeze=False)

        for i, img in enumerate(img_set):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.savefig(self.results_dir / Path('batch_' + str(file_idx) + '.png'), dpi=400)

    @torch.no_grad()
    def test_model(self, dataloader: torch.utils.data.DataLoader, results_dir: Path, no_visual: bool = True, no_save: bool = False, line_width: int = 3):
        
        self.results_dir = results_dir
        for idx, batch in enumerate(dataloader):
            
            # This preprocesses raw batch
            batch = convert_image_dtype(
                batch, dtype=torch.float32).to(self.device)
            
            # Feed input to model
            outputs = self.model(batch)
            
            # Build image with predicted boxes
            vis_result = [
                draw_bounding_boxes(
                    torch.tensor(img * 255).type(torch.ByteTensor), boxes=output['boxes'][output['scores'] > self.conf_threshold], width=line_width, colors=self.generate_pil_colors(output['labels'][output['scores'] > self.conf_threshold]))
                for img, output in zip(batch, outputs)
            ]
            
            if not no_visual:
                # Visualize result
                self.show(img_set=vis_result)

            if not no_save:
                # Export result
                self.export(img_set=vis_result, file_idx=idx)

class VisualTest():
    # TODO fix this class for xyxy and not xywhn
    def __init__(self, num_classes: int = 12):
        super().__init__()

        matplotlib.style.use('ggplot')
        plt.rcParams["savefig.bbox"] = 'tight'

        self.num_classes = num_classes
        self.colors = list(sorted(ImageColor.colormap.items()))[
            :self.num_classes]
    
    def generate_pil_colors(self, labels: List[int]):
        return [self.colors[sample][0] for sample in labels]

    def draw_bboxes(self, img: np.ndarray, boxes: torch.Tensor, labels: List[int], line_width: int = 3):
        return draw_bounding_boxes(img, boxes, colors=self.generate_pil_colors(labels), width=line_width)

    def show(self, img_set):
        if not isinstance(img_set, list):
            img_set = [img_set]

        fig, axs = plt.subplots(ncols=len(img_set), squeeze=False)

        for i, img in enumerate(img_set):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.show()
    
    def rescale_boxes(self, img: torch.ByteTensor, boxes: torch.Tensor):
        _, img_height, img_width = img.shape

        print(img_height, img_width)
        print(boxes)

        boxes[:, 0] = boxes[:, 0] * img_width
        boxes[:, 1] = boxes[:, 1] * img_height
        boxes[:, 2] = boxes[:, 2] * img_width
        boxes[:, 3] = boxes[:, 3] * img_height

        boxes[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2)
        boxes[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        boxes = torch.round(boxes)
        
        return boxes

    def visualize(self, img: np.ndarray, boxes: torch.Tensor, labels: List[int]):
        img = img.clone().detach().type(torch.ByteTensor)
        img_set = self.draw_bboxes(
            img=img, boxes=self.rescale_boxes(img=img, boxes=boxes), labels=labels)
        self.show(img_set=img_set)
