#!/usr/bin/env python3
"""
Fast Wan video generation with model preloading.
Models are loaded once and kept in memory for quick inference.
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool

# Global model cache
_MODEL_CACHE = {}

def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


class WanFastGenerator:
    """Fast Wan generator with model preloading"""
    
    def __init__(self, task="vace-1.3B", ckpt_dir=None, device_id=0):
        """
        Initialize and load models once
        
        Args:
            task: Task type (e.g., "vace-1.3B")
            ckpt_dir: Path to checkpoint directory
            device_id: GPU device ID
        """
        self.task = task
        self.ckpt_dir = ckpt_dir
        self.device_id = device_id
        self.cfg = WAN_CONFIGS[task]
        
        _init_logging()
        logging.info(f"Initializing WanFastGenerator for task: {task}")
        logging.info(f"Checkpoint directory: {ckpt_dir}")
        
        # Load model
        if "vace" in task:
            logging.info("Loading VACE pipeline (this may take a while)...")
            self.pipeline = wan.WanVace(
                config=self.cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device_id,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
            )
            logging.info("✓ VACE pipeline loaded successfully!")
        else:
            raise NotImplementedError(f"Task {task} not yet supported in fast mode")
    
    def generate(
        self,
        prompt,
        src_ref_images=None,
        save_file=None,
        size="832*480",
        frame_num=41,
        sample_steps=25,
        sample_shift=16.0,
        sample_solver='unipc',
        guide_scale=5.0,
        base_seed=-1,
        offload_model=True
    ):
        """
        Generate video quickly (models already loaded)
        
        Args:
            prompt: Text prompt for generation
            src_ref_images: Comma-separated list of reference image paths
            save_file: Path to save output video
            size: Video size (e.g., "832*480")
            frame_num: Number of frames
            sample_steps: Sampling steps
            sample_shift: Sampling shift
            sample_solver: Solver type
            guide_scale: Guidance scale
            base_seed: Random seed
            offload_model: Whether to offload model after generation
        
        Returns:
            Path to generated video
        """
        logging.info(f"Generating video with prompt: {prompt}")
        
        if base_seed < 0:
            base_seed = random.randint(0, sys.maxsize)
        
        # Prepare reference images if provided
        src_video = None
        src_mask = None
        ref_images_list = None
        
        if src_ref_images:
            ref_images_list = [src_ref_images.split(',')]
            logging.info(f"Reference images: {ref_images_list}")
        
        # Prepare source data
        src_video, src_mask, src_ref_images_tensor = self.pipeline.prepare_source(
            [src_video],
            [src_mask],
            ref_images_list,
            frame_num,
            SIZE_CONFIGS[size],
            self.device_id
        )
        
        # Generate video
        logging.info("Generating video...")
        video = self.pipeline.generate(
            prompt,
            src_video,
            src_mask,
            src_ref_images_tensor,
            size=SIZE_CONFIGS[size],
            frame_num=frame_num,
            shift=sample_shift,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=base_seed,
            offload_model=offload_model
        )
        
        # Save video
        if save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
            save_file = f"{self.task}_{size}_{formatted_prompt}_{formatted_time}.mp4"
        
        logging.info(f"Saving generated video to {save_file}")
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        logging.info("✓ Video generation complete!")
        return save_file


def get_generator(task="vace-1.3B", ckpt_dir=None, device_id=0):
    """
    Get or create a cached generator instance.
    Models are loaded once and reused.
    
    Args:
        task: Task type
        ckpt_dir: Checkpoint directory
        device_id: GPU device ID
        
    Returns:
        WanFastGenerator instance
    """
    cache_key = f"{task}_{ckpt_dir}_{device_id}"
    
    if cache_key not in _MODEL_CACHE:
        logging.info(f"Creating new generator (first run - will be slow)")
        _MODEL_CACHE[cache_key] = WanFastGenerator(
            task=task,
            ckpt_dir=ckpt_dir,
            device_id=device_id
        )
    else:
        logging.info(f"Using cached generator (fast!)")
    
    return _MODEL_CACHE[cache_key]


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Fast Wan video generation with model preloading"
    )
    parser.add_argument("--task", type=str, default="vace-1.3B")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--src_ref_images", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=41)
    parser.add_argument("--sample_steps", type=int, default=25)
    parser.add_argument("--sample_shift", type=float, default=16.0)
    parser.add_argument("--sample_solver", type=str, default='unipc')
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    
    args = parser.parse_args()
    
    # Get or create generator (models loaded once)
    generator = get_generator(
        task=args.task,
        ckpt_dir=args.ckpt_dir,
        device_id=args.device_id
    )
    
    # Generate video (fast!)
    output_file = generator.generate(
        prompt=args.prompt,
        src_ref_images=args.src_ref_images,
        save_file=args.save_file,
        size=args.size,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        sample_solver=args.sample_solver,
        guide_scale=args.sample_guide_scale,
        base_seed=args.base_seed,
        offload_model=args.offload_model
    )
    
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()

