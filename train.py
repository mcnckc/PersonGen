import warnings
import gc
import random
from datetime import datetime
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from omegaconf import open_dict

from src.constants.trainer import TRAINER_NAME_TO_CLASS
from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.nb_utils.eval_sets import evaluation_sets
from src.metrics.global_tracker import GlobalTracker

warnings.filterwarnings("ignore", category=UserWarning)


def train(config, device, logger, writer, train_reward_model,
          val_reward_models, multi_prompt=False, global_tracker=None):
    

    # build stable diffusion models
    start_time = datetime.now()
    model = instantiate(config.model).to(device)
    if model.use_ema:
        model.ema_unet.to(device)

    # build reward models

    all_models_with_tokenizer = val_reward_models + [model, train_reward_model]

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(
        config,
        device=device,
        all_models_with_tokenizer=all_models_with_tokenizer,
    )

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    scaler = GradScaler()

    epoch_len = config.trainer.get("epoch_len")

    if config.trainer.type not in TRAINER_NAME_TO_CLASS:
        raise ValueError(f"Trainer type must be one of {TRAINER_NAME_TO_CLASS}")

    trainer_cls = TRAINER_NAME_TO_CLASS[config.trainer.type]
    trainer = trainer_cls(
        model=model,
        train_reward_model=train_reward_model,
        val_reward_models=val_reward_models,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        multi_prompt=multi_prompt,
        global_tracker=global_tracker
    )
    print("train preparation took", (datetime.now() - start_time).total_seconds())
    start_time = datetime.now()
    trainer.train()
    print("train itself took", (datetime.now() - start_time).total_seconds())

@hydra.main(version_base=None, config_path="src/configs", config_name="refl_train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    #torch.cuda.memory._record_memory_history()
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
    """
    if True:
        set_random_seed(config.trainer.seed)

        project_config = OmegaConf.to_container(config)
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
        if config.trainer.device == "auto":
            print("Auto")
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            print(f"Device is {config.trainer.device}")
            device = config.trainer.device
            print(device)

        with open_dict(config.reward_models.train_model):
            config.reward_models.train_model = OmegaConf.merge(config.reward_models.train_model,
                        {"target_prompt": config.datasets.train.target_prompt})
        print("copied prompt:", config.reward_models.train_model.target_prompt)
        train_reward_model = instantiate(
            config.reward_models["train_model"], device=device
        ).to(device)
        train_reward_model.requires_grad_(False)

        val_reward_models = []


        if config.trainer.multi_prompt:
            prompts = [p for group in config.trainer.prompt_groups for p in evaluation_sets[group]]
            if config.trainer.max_prompts >= 0:
                random.shuffle(prompts)
                prompts = prompts[:config.trainer.max_prompts]
            fill_in = config.reward_models.train_model.placeholder_token + ' ' + \
            config.reward_models.train_model.class_name
            print("FILL IN FOR PROMPTS:", fill_in)
            prompts = [p.format(fill_in) for p in prompts]
            print("ALL PROMPTS:", prompts)
            global_tracker = GlobalTracker(device, prompts, writer=writer)
            for prompt_id, prompt in enumerate(prompts):
                print("START PROMPT ID", prompt_id, prompt)
                start_time = datetime.now()
                train_reward_model.update_target_prompt(prompt)
                global_tracker.set_prompt(prompt_id)
                with open_dict(config.datasets):
                    config.datasets.train = OmegaConf.merge(config.datasets.train,
                                                            {"target_prompt":prompt})
                    config.datasets.val = OmegaConf.merge(config.datasets.val,
                                                                {"target_prompt":prompt})
                print("Pre train time:", (datetime.now() - start_time).total_seconds())
                train(config, device, logger, writer, train_reward_model, val_reward_models, True, global_tracker)
                gc.collect()
                torch.cuda.empty_cache()
                print("LOGGING:", (datetime.now() - start_time).total_seconds())
                writer.exp.log_metrics({
                    "Time for one prompt": (datetime.now() - start_time).total_seconds(),
                }, step=prompt_id)
            #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print("FINISHED ALL PROMPTS")
            start_time = datetime.now()
            train_reward_model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            #print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    #torch.cuda.memory._dump_snapshot(f"memory-{prompt_id}.pickle")
            for reward_model_config in config.reward_models["val_models"]:
                with open_dict(reward_model_config.config):
                    reward_model_config.config = OmegaConf.merge(reward_model_config.config, 
                                                            {"target_prompt": config.datasets.train.target_prompt})
                reward_model = instantiate(reward_model_config, device=device).to(device)
                reward_model.requires_grad_(False)
                val_reward_models.append(reward_model)
            val_reward_models[0].zero_time_stats()

            global_tracker.score_val_images(val_reward_models[0])
            writer.exp.log_metrics({
                "Total validation time": (datetime.now() - start_time).total_seconds(),
            }, step=0)
            writer.exp.log_metrics({
                "Total validation time per image": (datetime.now() - start_time).total_seconds() / ((config.trainer.n_epochs + 1) * len(prompts)),
            }, step=0)
            writer.exp.log_metrics({
                "Total pf calls": val_reward_models[0].db.pf_calls,
            }, step=0)
            writer.exp.log_metrics({
                "Total cp calls": val_reward_models[0].db.cp_calls,
            }, step=0)
            writer.exp.log_metrics({
                "Total clean pf time": val_reward_models[0].db.pf_clean_time,
            }, step=0)
            writer.exp.log_metrics({
                "Total clean cp time": val_reward_models[0].db.cp_clean_time,
            }, step=0)
            global_tracker.log_total()
        else:
            train(config, device, logger, writer, train_reward_model, val_reward_models)
        print("Finished, stopping profiler")
        #prof.stop()
    
    #prof.export_memory_timeline(f"memory2.html", device="cuda:0")

    

import clip

if __name__ == "__main__":
    MODEL_NAME = "ViT-B/32"
    model, transform = clip.load(MODEL_NAME, device="cuda", jit=False)
    #main()
