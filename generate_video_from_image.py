#!/usr/bin/env python3
"""
Script to generate a video from an existing image and its reasoning trace.
Uses Anthropic's Claude to create an optimized video generation prompt.
"""

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path
import anthropic


def encode_image_to_base64(image_path: str) -> str:
    """Read image and encode to base64."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def read_reasoning_file(text_path: str) -> dict:
    """Read the reasoning trace file and extract reasoning and original prompt."""
    with open(text_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by the reasoning end marker
    parts = content.split("--- END OF REASONING ---")
    
    if len(parts) >= 2:
        reasoning = parts[0].strip()
        # Extract the final prompt (after the reasoning)
        remaining = parts[1].strip()
        lines = remaining.split("\n")
        # Find the actual prompt (usually after "Generating image with prompt:")
        original_prompt = ""
        for i, line in enumerate(lines):
            if "Generating image with prompt:" in line or line.strip().startswith("Cartoon"):
                # Get the prompt from the next non-empty lines
                prompt_lines = []
                for j in range(i, len(lines)):
                    if lines[j].strip():
                        prompt_lines.append(lines[j].strip())
                original_prompt = " ".join(prompt_lines).replace("Generating image with prompt:", "").strip()
                break
        
        if not original_prompt:
            # Fallback: use everything after reasoning
            original_prompt = remaining
    else:
        # No marker found, try to split differently
        reasoning = content[:len(content)//2]
        original_prompt = content[len(content)//2:]
    
    return {
        "reasoning": reasoning,
        "original_prompt": original_prompt
    }


def generate_video_prompt_with_claude(
    image_path: str,
    reasoning: str,
    original_prompt: str,
    api_key: str
) -> str:
    """Use Claude to generate an optimized video generation prompt."""
    
    # Determine image media type
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = media_type_map.get(ext, "image/png")
    
    # Encode image
    image_data = encode_image_to_base64(image_path)
    
    # Create the prompt for Claude
    system_prompt = """You are an expert in video generation and creative direction. Your task is to take a static image and its creation reasoning, and generate an engaging video generation prompt that brings the image to life with motion, dynamics, and cinematic appeal.

Key principles:
1. Preserve the core visual elements and style from the original image
2. Add motion, camera movements, and dynamic elements appropriate for video
3. Consider cinematography: camera angles, movements (pan, zoom, tracking), lighting changes
4. Add temporal elements: how things change over time, sequences of actions
5. Maintain the tone and atmosphere of the original
6. Keep the prompt concise but detailed (50-100 words)
7. Make it suitable for state-of-the-art video generation models"""

    user_prompt = f"""I have a generated image and want to create an interesting video from it.

ORIGINAL IMAGE CREATION REASONING:
{reasoning}

ORIGINAL IMAGE PROMPT:
{original_prompt}

Please analyze the image and create an optimized video generation prompt that:
1. Captures the essence and style of the static image
2. Adds appropriate motion and camera dynamics to make it cinematic
3. Maintains visual consistency with the original
4. Creates an engaging 3-5 second video narrative
5. Is optimized for the Wan 2.1 video generation model

OUTPUT FORMAT:
Provide ONLY the video generation prompt, nothing else. No explanations, no preamble, just the prompt text."""

    # Call Claude API
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ],
            }
        ],
    )
    
    video_prompt = message.content[0].text.strip()
    return video_prompt


def generate_video(
    prompt: str,
    task: str = "t2v-14B",
    size: str = "1280*720",
    ckpt_dir: str = "./Wan2.1-T2V-14B",
    output_dir: str = None,
    additional_args: list = None
):
    """Call generate.py to create the video."""
    
    cmd = [
        sys.executable,
        "generate.py",
        "--task", task,
        "--size", size,
        "--ckpt_dir", ckpt_dir,
        "--prompt", prompt
    ]
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    
    if additional_args:
        cmd.extend(additional_args)
    
    print("\n" + "="*80)
    print("EXECUTING VIDEO GENERATION COMMAND:")
    print(" ".join(cmd))
    print("="*80 + "\n")
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"\n✗ Video generation failed with return code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print("\n✓ Video generation completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video from an image and its reasoning trace using Claude."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="video_gen_prompts/generated_image.png",
        help="Path to the generated image"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="video_gen_prompts/prompt+reasoning.txt",
        help="Path to the text file containing reasoning and prompt"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        help="Video generation task (default: t2v-14B)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        help="Video size (default: 1280*720)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./Wan2.1-T2V-14B",
        help="Checkpoint directory (default: ./Wan2.1-T2V-14B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated video"
    )
    parser.add_argument(
        "--skip-claude",
        action="store_true",
        help="Skip Claude API call and use original prompt directly"
    )
    
    args, unknown_args = parser.parse_known_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"✗ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.text):
        print(f"✗ Error: Text file not found: {args.text}")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.skip_claude:
        print("✗ Error: Anthropic API key required. Set --api-key or ANTHROPIC_API_KEY env var")
        print("   Or use --skip-claude to use the original prompt without Claude enhancement")
        sys.exit(1)
    
    print("="*80)
    print("VIDEO GENERATION FROM IMAGE AND REASONING")
    print("="*80)
    print(f"\n📸 Image: {args.image}")
    print(f"📝 Text: {args.text}")
    print(f"🎬 Task: {args.task}")
    print(f"📐 Size: {args.size}")
    
    # Read reasoning and original prompt
    print("\n" + "-"*80)
    print("READING REASONING AND ORIGINAL PROMPT...")
    print("-"*80)
    
    data = read_reasoning_file(args.text)
    reasoning = data["reasoning"]
    original_prompt = data["original_prompt"]
    
    print(f"\n📖 Original prompt length: {len(original_prompt)} characters")
    print(f"🧠 Reasoning length: {len(reasoning)} characters")
    print(f"\n📖 Original prompt:\n{original_prompt}")
    
    # Generate video prompt with Claude
    if args.skip_claude:
        print("\n⏭️  Skipping Claude - using original prompt for video generation")
        video_prompt = original_prompt
    else:
        print("\n" + "-"*80)
        print("GENERATING VIDEO PROMPT WITH CLAUDE...")
        print("-"*80)
        
        try:
            video_prompt = generate_video_prompt_with_claude(
                args.image,
                reasoning,
                original_prompt,
                api_key
            )
            print(f"\n✓ Claude generated video prompt:\n")
            print("="*80)
            print(video_prompt)
            print("="*80)
        except Exception as e:
            print(f"\n✗ Error calling Claude API: {e}")
            print("Falling back to original prompt...")
            video_prompt = original_prompt
    
    # Generate the video
    print("\n" + "-"*80)
    print("GENERATING VIDEO...")
    print("-"*80)
    
    # generate_video(
    #     prompt=video_prompt,
    #     task=args.task,
    #     size=args.size,
    #     ckpt_dir=args.ckpt_dir,
    #     output_dir=args.output_dir,
    #     additional_args=unknown_args
    # )
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

