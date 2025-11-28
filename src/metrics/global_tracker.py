import torch
from torchvision.transforms import functional as F
from src.metrics.tracker import MetricTracker

class GlobalTracker:
    """
    Class to aggregate metrics from many train runs for different prompts.
    """

    def __init__(self, prompts, writer=None):
        self.prompts = prompts
        self.prompt_id = 0
        self.metrics = [{} for _ in range(len(self.prompts))]
        self.val_metrics = [{} for _ in range(len(self.prompts))]
        self.writer = writer
    
    def set_prompt(self, id: int):
        self.prompt_id = id

    def update(self, metric_tracker: MetricTracker, batch, step: int, val: bool = False):
        if val:
            self.val_metrics[self.prompt_id][step] = {name: metric_tracker.avg(name) for name in metric_tracker.keys()}
        else:
            self.metrics[self.prompt_id][step] = {name: metric_tracker.avg(name) for name in metric_tracker.keys()}
        self.writer.exp.log_image(
                image_data=F.to_pil_image(batch["image"][0].to(torch.float16).cpu()), 
                name=self.prompts[self.prompt_id], step=step
        )
    
    def log_total(self):
        for step in self.metrics[0].keys():
            self.writer.exp.log_metrics({
                    name + '_train': sum(self.metrics[i][step][name] for i in range(len(self.metrics))) 
                    for name in self.metrics[0][step].keys()
                },
                step=step
            )
        for step in self.val_metrics[0].keys():
            self.writer.exp.log_metrics({
                    name + '_val': sum(self.val_metrics[i][step][name] for i in range(len(self.val_metrics))) 
                    for name in self.val_metrics[0][step].keys()
                },
                step=step
            )

    
