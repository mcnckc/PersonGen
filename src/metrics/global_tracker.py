import torch
from datetime import datetime
from torchvision.transforms import functional as F
from src.metrics.tracker import MetricTracker

class GlobalTracker:
    """
    Class to aggregate metrics from many train runs for different prompts.
    """

    def __init__(self, device, prompts, writer=None):
        self.device = device
        self.prompts = prompts
        self.prompt_id = 0
        self.metrics = [{} for _ in range(len(self.prompts))]
        self.val_images = [{} for _ in range(len(self.prompts))]
        self.writer = writer
    
    def set_prompt(self, id: int):
        self.prompt_id = id

    def update(self, metric_tracker: MetricTracker, batch, step: int, val: bool = False):
        if val:
            self.val_images[self.prompt_id][step] = batch["image"].to(torch.float16).detach().cpu()
        else:
            self.metrics[self.prompt_id][step] = {name: metric_tracker.avg(name) for name in metric_tracker.keys()}
            print("LEAK?", self.metrics[self.prompt_id][step])
            self.writer.exp.log_image(
                    image_data=F.to_pil_image(batch["image"][0].to(torch.float16).cpu()), 
                    name=self.prompts[self.prompt_id], step=step
            )

    def score_val_images(self, reward_model):
        self.val_metrics = [{} for _ in range(len(self.prompts))]
        num_steps = len(self.val_images[0])
        for pid, prompt in enumerate(self.prompts):
            reward_model.update_target_prompt(prompt)
            step_id = 0
            for step, image in self.val_images[pid].items():
                start_time = datetime.now()
                batch = {"image":image.to(self.device)}
                reward_model.score(batch)
                print(f"SCORING BATCH: of size {len(batch["image"])}", batch)
                self.val_metrics[pid][step] = {loss_name: loss for loss_name, loss in batch.items() if loss_name != "image"}
                self.writer.exp.log_metrics({
                        "One batch validation time": (datetime.now() - start_time).total_seconds(),
                }, step=pid * num_steps + step_id)
                step_id += 1

    def log_total(self):
        for pid in range(len(self.metrics)):
            print(f"LOG TOTAL FOR {pid}\n", self.metrics[pid])
        for step in self.metrics[0].keys():
            self.writer.exp.log_metrics({
                    name + '_train': sum(self.metrics[i][step][name] for i in range(len(self.metrics))) / len(self.metrics) 
                    for name in self.metrics[0][step].keys()
                },
                step=step
            )
        for step in self.val_metrics[0].keys():
            self.writer.exp.log_metrics({
                    name + '_val': sum(self.val_metrics[i][step][name] for i in range(len(self.val_metrics))) / len(self.val_metrics)
                    for name in self.val_metrics[0][step].keys()
                },
                step=step
            )

    
