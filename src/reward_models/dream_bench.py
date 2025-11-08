"""Based on https://github.com/rinongal/textual_inversion/blob/main/evaluation/clip_eval.py"""
import typing as tp
import os
import warnings
import requests
from typing import List, Optional, Dict, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.transforms import functional as F

import regex
import tqdm.autonotebook as tqdm

import clip

import numpy as np

import torch
import torch.backends.cuda

from torchvision import transforms

from transformers import AutoProcessor, GenerationConfig, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

import PIL
from PIL.Image import Image

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

CLIP_MODEL_NAME = "ViT-B/32"

class CLIPEvaluator(object):
    def __init__(self, device, clip_model=CLIP_MODEL_NAME) -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-1.0, -1.0, -1.0],
                    std=[2.0, 2.0, 2.0])
            ] +                               # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1]
            clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
            clip_preprocess.transforms[4:]    # + skip convert PIL to tensor
        )
        print(self.preprocess)
        print(clip_preprocess)
        for transform in self.preprocess.transforms:
            if isinstance(transform, transforms.Resize):
                transform.antialias = False

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)
    
    @torch.no_grad()
    def encode_pil_images(self, images: torch.Tensor) -> torch.Tensor:
        images = torch.stack([self.clip_preprocess(image).to(self.device) for image in images], dim=0)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def get_pil_image_features(self, imgs: List[Image], norm: bool = True) -> torch.Tensor:
        image_features = self.encode_pil_images(imgs)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        similarity_matrix = src_img_features @ gen_img_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()

    def txt_to_img_similarity(self, text, generated_images):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        similarity_matrix = text_features @ gen_img_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()


class DINOEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model=CLIP_MODEL_NAME, dino_model='dinov2_vits14') -> None:
        super().__init__(device, clip_model=clip_model)

        self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_model).to(self.device)

        self.dino_preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @torch.no_grad()
    def dino_encode_images(self, images: List[Image]) -> torch.Tensor:
        images = torch.stack([self.dino_preprocess(image) for image in images])
        images = images.to(self.device)
        return self.dino_model(images)

    def get_dino_image_features(self, img: List[Image], norm: bool = True) -> torch.Tensor:
        image_features = self.dino_encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features


class ExpEvaluator:
    def __init__(self, device, config):
        self.device = device
        self.evaluator = DINOEvaluator(device=device)
        self.train_images = self.load_images(config.src_img_dir)
        self.train_images_features, self.dino_train_images_features = self._get_image_features(self.train_images)
        self.global_results = {}
        self.global_results['real_image_similarity'], self.global_results['real_image_similarity_mx'] = (
            self._calc_self_similarity(self.train_images_features)
        )
        self.global_results['dino_real_image_similarity'], self.global_results['dino_real_image_similarity_mx'] = (
            self._calc_self_similarity(self.dino_train_images_features)
        )

        

        clean_label = (
                config.target_prompt
                .replace('{0} {1}'.format(config.placeholder_token, config.class_name), '{0}')
                .replace('{0}'.format(config.placeholder_token), '{0}')
            )
        empty_label = clean_label.replace('{0} ', '').replace(' {0}', '')
        self.empty_label_with_class = clean_label.format(config.class_name)
        self.global_results['target_class_prompt'] = self.empty_label_with_class
        self.global_results['target_clean_prompt'] = empty_label
        self.empty_label_features = self.evaluator.get_text_features(empty_label)
        self.empty_label_with_class_features = self.evaluator.get_text_features(self.empty_label_with_class)
        

    def load_images(self, src_img_dir):
        src_image_paths = []
        src_images = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for filename in os.listdir(src_img_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                src_image_paths.append(os.path.join(src_img_dir, filename))
        if not src_image_paths:
            raise ValueError(f"Directory {src_img_dir} has no images")
        for img_path in src_image_paths:
            src_images.append(PIL.Image.open(img_path))
        return src_images
    
    @staticmethod
    def _images_to_tensor(images):
        """
        Convert list of numpy.ndarray images with numpy.uint8 encoding ([0, 255] range)
            to torch.Tensor with torch.float32 encoding ([-1.0, 1.0] range)
        """
        images = np.stack(images)
        images = torch.from_numpy(np.transpose(images, axes=(0, 3, 1, 2)))
        return torch.clamp(images / 127.5 - 1.0, min=-1.0, max=1.0)

    @staticmethod
    def _calc_similarity(left_features, right_features):
        similarity_matrix = left_features @ right_features.T
        return similarity_matrix.mean().item(), similarity_matrix.cpu().numpy().tolist()

    @staticmethod
    def _calc_self_similarity(features):
        similarity_matrix = features @ features.T
        return similarity_matrix[torch.triu(
            torch.ones_like(similarity_matrix, dtype=bool), diagonal=1
            )].mean().item(), similarity_matrix.cpu().numpy().tolist()
    
    @torch.no_grad()
    def _get_image_features(self, PIL_images, resolution=None):
        # noinspection PyPep8Naming
        #print(type(images[0]))
        #print(images)
        
        

        dino_images_features = self.evaluator.get_dino_image_features(PIL_images)

        images_features = self.evaluator.get_pil_image_features(PIL_images)

        return images_features, dino_images_features

    @torch.no_grad()
    def __call__(self, PIL_images, verbose=False):
        results = {}

        images_features, dino_images_features = self._get_image_features(PIL_images)

        results['image_similarities'], results['image_similarities_mx'] = (
            self._calc_similarity(self.train_images_features, images_features)
        )
        results['dino_image_similarities'], results['dino_image_similarities_mx'] = (
            self._calc_similarity(self.dino_train_images_features, dino_images_features)
        )
        if verbose:
            print('IS: {0:.3f} ({1:.3f})'.format(
                results['image_similarities'], results['dino_image_similarities'],
            ))


        (results['text_similarities'], results['text_similarities_mx']) = \
        self._calc_similarity(self.empty_label_features, images_features)

        

        (results['text_similarities_with_class'],results['text_similarities_mx_with_class'],) = \
        self._calc_similarity(self.empty_label_with_class_features, images_features)
        
        
        results |= self.global_results
        if verbose:
            print('TS: {0:.3f} {1}'.format(
                results['text_similarities'],
                results['target_clean_prompt']
            ))
            print('TS: {0:.3f} {1}'.format(
                results['text_similarities_with_class'],
                results['target_class_prompt']
            ))

        return results


class DreamBenchPPEvaluator(ExpEvaluator):
    _PROMPTS = {
        'PF_gpt_prompt': 'gpt_prompt_text_full.txt',
        'PF_user_prompt': 'user_prompt_text_full.txt',

        'CP_gpt_prompt': 'gpt_prompt_subject_full.txt',
        'CP_user_prompt': 'user_prompt_subject_full.txt',
    }
    _PROMPTS_URL = 'https://raw.githubusercontent.com/yuangpeng/dreambench_plus/refs/heads/main/dreambench_plus/prompts/'
    
    def __init__(
        self, device,
        config, 
        vllm_model='Qwen/Qwen3-VL-30B-A3B-Instruct', 
        prompts_path=None,
        determenistic=True
    ) -> None:
        super().__init__(device, config)
        
        device_idx = torch.device(device).index
        old_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' if device_idx is None else str(device_idx)
        
        self.llm_model  = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        
        if determenistic:
            # Deterministic sampling: Qwen3-VL-30B-A3B-Instruct-FP8 vLLM example
            self.sampling_params = GenerationConfig(
                temperature=0,
                top_k=-1,
                max_tokens=1024
            )
        else:
            # Qwen3 non-thinking or Qwen3-VL-30B-A3B-Instruct generation_config.json
            self.sampling_params = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                presence_penalty=1.5,
                max_tokens=32768
            )
        
        self.processor = AutoProcessor.from_pretrained(vllm_model)

        self.prompts = {}
        for key, file_name in self._PROMPTS.items():
            if prompts_path is not None:
                with open(os.path.join(prompts_path, file_name)) as file:
                    self.prompts[key] = file.read().strip()
            else:
                self.prompts[key] = requests.request(
                    'get', os.path.join(self._PROMPTS_URL, file_name)
                ).content.decode()

        if old_visible_devices is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible_devices
        
        self.global_results = {}
        CP_to_process = []

        for i, left_image in enumerate(self.train_images):
            for right_image in self.train_images[i:]:
                CP_to_process.append((left_image, right_image))
        self.global_results['real_CP_mx'] = np.zeros((len(self.train_images), len(self.train_images)), dtype=float)

        source_images, CP_target_images = [*zip(*CP_to_process)]

        chunk_size = 5000
        CPs_all = [], []
        if 'seeds' not in config:
            config['seeds'] = (179, )

        for seed in config.seeds:
            CPs = []
                
            for source_images_batch, CP_target_images_batch in zip(
                [source_images[idx:idx + chunk_size] for idx in range(0, len(source_images), chunk_size)],
                [CP_target_images[idx:idx + chunk_size] for idx in range(0, len(CP_target_images), chunk_size)]
            ):
                CPs += self.get_concept_preservation(source_images_batch, CP_target_images_batch, return_texts=False, seed=seed)

            CPs_all.append(np.array(CPs, dtype=float))
        
        # If all evaluations failed then NaN will cause warnings - ignore them
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            CPs_all = np.nanmean(CPs_all, axis=0)
            
        # Set all NaN scores to 0.0
        CPs_all = np.nan_to_num(CPs_all, nan=0.0)
        
        CP_idx = 0
        for idx, _ in enumerate(self.train_images):
            for jdx in range(idx, len(self.train_images)):
                self.global_results['real_CP_mx'][idx, jdx] = CPs_all[CP_idx]
                CP_idx += 1
                
        self.global_results['real_CP'] = \
        self.global_results['real_CP_mx'][torch.triu(
            torch.ones_like(self.global_results['real_CP_mx'], dtype=bool), diagonal=1
            )].mean().item()
        
        self.global_results['real_CP_mx'] = self.global_results['real_CP_mx'].tolist()
        

    def _process_messagess(self, messages):
        text_input = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
            
        inputs = {
            'prompt': text_input,
            'multi_modal_data': {},
            'mm_processor_kwargs': video_kwargs
        }
        if image_inputs is not None:
            inputs['multi_modal_data']['image'] = image_inputs
        if video_inputs is not None:
            inputs['multi_modal_data']['video'] = video_inputs

        return inputs
    
    def _process_messages_parallel(self, all_messages: List, max_workers: int = 32):
        results = [None] * len(all_messages)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_messagess, messages): idx
                for idx, messages in enumerate(all_messages)
            }
            for future in tqdm.tqdm(as_completed(future_to_index), desc='Processing messages', total=len(all_messages)):
                idx = future_to_index[future]
                results[idx] = future.result()
                    
        return results
        
    def _get_PF_inputs(self, prompts_batch: List[str], target_images_batch: List[Image]):
        all_messages = []
        for prompt, target_image in tqdm.tqdm(zip(prompts_batch, target_images_batch), desc='Preparing PF inputs'):
            messages = [
                {"role": "user", "content": [{"type": "text", "text": self.prompts['PF_user_prompt']}]},
                {"role": "assistant", "content": [{"type": "text", "text": self.prompts['PF_gpt_prompt']}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": target_image},
                    ],
                },
            ]
            all_messages.append(messages)
        return self._process_messages_parallel(all_messages)

    def _get_CP_inputs(self, source_images_batch: List[Image], target_images_batch: List[Image]):
        all_messages = []
        for source_image, target_image in tqdm.tqdm(zip(source_images_batch, target_images_batch), desc='Preparing CP inputs'):
            messages = [
                {"role": "user", "content": [{"type": "text", "text": self.prompts['CP_user_prompt']}]},
                {"role": "assistant", "content": [{"type": "text", "text": self.prompts['CP_gpt_prompt']}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source_image},
                        {"type": "image", "image": target_image},
                    ],
                },
            ]
            all_messages.append(messages)
        return self._process_messages_parallel(all_messages)

    @staticmethod
    def _get_score(responce: str) -> Optional[int]:
        responce = responce.replace('**', '')
        try:
            pattern = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"
            scores = [
                int(score) for _, score in 
                regex.findall(pattern, responce)
            ]
            return scores[0]
        except Exception:
            return None
    
    def get_prompt_following(
        self, prompts_batch: List[str], target_images_batch: List[Image], return_texts: bool = False, seed=6417,
    ) -> Union[List[Optional[int]], Tuple[List[Optional[int]], List[str]]]:
        PF_inputs = self._get_PF_inputs(prompts_batch, target_images_batch)
        print(f"PF inputs:{PF_inputs}")
        sampling_params = self.sampling_params.clone()
        sampling_params.seed = seed
        PF_responses_ids = self.llm_model.generate(PF_inputs, generation_config=sampling_params)
        PF_responses_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(PF_inputs, PF_responses_ids)
        ]
        PF_texts = self.processor.batch_decode(
            PF_responses_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(PF_texts)
        #PF_texts = [response.outputs[0].text for response in PF_responses]
        PF_scores = [self._get_score(text) for text in PF_texts]

        if return_texts:
            return PF_scores, PF_texts
        return PF_scores

    def get_concept_preservation(
        self, source_images_batch: List[Image], target_images_batch: List[Image], return_texts: bool = False, seed=6417,
    ) -> Union[List[Optional[int]], Tuple[List[Optional[int]], List[str]]]:
        CP_inputs = self._get_CP_inputs(source_images_batch, target_images_batch)
        
        sampling_params = self.sampling_params.clone()
        sampling_params.seed = seed
        CP_responses_ids = self.llm_model.generate(CP_inputs, generation_config=sampling_params)
        CP_responses_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(CP_inputs, CP_responses_ids)
        ]
        CP_texts = self.processor.batch_decode(
            CP_responses_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(CP_texts)
        #CP_texts = [response.outputs[0].text for response in CP_responses]
        CP_scores = [self._get_score(text) for text in CP_texts]
        
        if return_texts:
            return CP_scores, CP_texts
        return CP_scores
    
    @torch.no_grad()
    def __call__(self, images, verbose=False, seeds=(179, )):  
        base_results = super().__call__(images, verbose=verbose)
        base_results |= self.global_results
        CP_to_process, PF_to_process = [], []
        
        results = {}

        for train_image in self.train_images:
            for image in images:
                CP_to_process.append((train_image, image))
        results['CPs_mx'] = np.zeros((len(self.train_images), len(images)), dtype=float)
            

        results['PFs_with_class_prompt'] = self.empty_label_with_class
            
        for image in images:
            PF_to_process.append((self.empty_label_with_class, image))
        results['PFs_mx_with_class'] = np.zeros((1, len(images)), dtype=float)

        prompts, PF_target_images = [*zip(*PF_to_process)]
        source_images, CP_target_images = [*zip(*CP_to_process)]

        chunk_size = 5000
        PFs_all, CPs_all = [], []
        for seed in seeds:
            CPs, PFs = [], []
            for prompts_batch, PF_target_images_batch in zip(
                [prompts[idx:idx + chunk_size] for idx in range(0, len(prompts), chunk_size)],
                [PF_target_images[idx:idx + chunk_size] for idx in range(0, len(PF_target_images), chunk_size)]
            ):
                PFs += self.get_prompt_following(prompts_batch, PF_target_images_batch, return_texts=False, seed=seed)
                
            for source_images_batch, CP_target_images_batch in zip(
                [source_images[idx:idx + chunk_size] for idx in range(0, len(source_images), chunk_size)],
                [CP_target_images[idx:idx + chunk_size] for idx in range(0, len(CP_target_images), chunk_size)]
            ):
                CPs += self.get_concept_preservation(source_images_batch, CP_target_images_batch, return_texts=False, seed=seed)

            CPs_all.append(np.array(CPs, dtype=float))
            PFs_all.append(np.array(PFs, dtype=float))
        
        # If all evaluations failed then NaN will cause warnings - ignore them
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            PFs_all = np.nanmean(PFs_all, axis=0)
            CPs_all = np.nanmean(CPs_all, axis=0)
            
        # Set all NaN scores to 0.0
        PFs_all = np.nan_to_num(PFs_all, nan=0.0)
        CPs_all = np.nan_to_num(CPs_all, nan=0.0)
        
        if verbose:
            print('Real Concept Preservation: {0:.3f}'.format(
                self.results['real_CP']
            ))

        

        for idx, _ in enumerate(self.train_images):
            for jdx, _ in enumerate(images):
                results['CPs_mx'][idx, jdx] = CPs_all[CP_idx]
                CP_idx += 1
                    
        results['CPs'] = results['CPs_mx'].mean().item()
            
        results['CPs_mx'] = results['CPs_mx'].tolist()
        if verbose:
            print('CP: {0:.3f}'.format(
                results['CPs'],
            ))

        for idx, _ in enumerate(images):
            results['PFs_mx_with_class'][0, idx] = PFs_all[PF_idx]
            PF_idx += 1
                
        results['PFs_with_class'] = results['PFs_mx_with_class'].mean().item()

        results['PFs_mx_with_class'] = results['PFs_mx_with_class'].tolist()
        if verbose:
            print('PF: {0:.3f} {1}'.format(
                results['PFs_with_class'],
                results['PFs_with_class_prompt']
            ))

        return base_results | results


def narrow_similarities(
        text_similarities: Dict[str, float], prompts: List[str], holder: Optional[str], verbose: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """ Takes dictionary of similarities (prompt -> value) and aggregates only values from the list of prompts
    :param text_similarities: dictionary of text similarities
    :param prompts: subset of templated keys (i.e. "a photo of {0}") from dictionary of text similarities
    :param holder: target holder (i.e. placeholder or placeholder with class name)
    :param verbose: if True then print existing errors
    :return: mean and std text similarity over list of selected prompts. If some prompts are missing then returns None
    """
    try:
        result = []
        for prompt in prompts:
            result.append(text_similarities[f"{prompt.format(holder)}"])

        return float(np.mean(result)), float(np.std(result))
    except Exception as ex:
        if verbose:
            print(f'Exception in narrow_similarities: {ex}')
        return None, None
    
MODEL_SUFFIX = "DreamBench"

class DreamBench(BaseModel):
    def __init__(self, device: torch.device, config):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )
        self.device = device
        self.db = DreamBenchPPEvaluator(device, config, prompts_path=config.prompts_path)
        self.tracked_metrics = {
            'source_ClipI':'real_image_similarity',
            'source_Dino':'dino_real_image_similarity',
            'source_CP':'real_CP',
            'ClipI':'image_similarities',
            'Dino':'dino_image_similarities',
            'CP':'CPs',
            'ClipT':'text_similarities_with_class',
            'PF':'PFs_with_class'
        }

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = clip.tokenize(
            caption,
            truncate=True,
        )

        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption
        }
    
    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        image = F.to_pil_image(batch["image"].squeeze(dim=0).to(torch.float32).cpu())
        results = self.db([image])
        batch.update({name1:results[name2] for name1, name2 in self.tracked_metrics.items()})
        return batch['CP'] + batch['PF']

    

