# Generate Video from Image & Reasoning

This script takes a generated image and its reasoning trace, uses Claude (Anthropic's LLM) to create an optimized video generation prompt, and then generates an engaging video.

## Prerequisites

1. **Anthropic API Key**: Get one from [https://console.anthropic.com/](https://console.anthropic.com/)
2. **Wan 2.1 Model**: Ensure you have the video generation model downloaded
3. **Python packages**: The script will auto-install `anthropic` if needed

## Quick Start

### Basic Usage

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Generate video from default files
python generate_video_from_image.py
```

This uses the default files:
- Image: `video_gen_prompts/generated_image.png`
- Text: `video_gen_prompts/prompt+reasoning.txt`

### Custom Files

```bash
python generate_video_from_image.py \
  --image path/to/your/image.png \
  --text path/to/your/reasoning.txt \
  --api-key "your-api-key"
```

### Custom Video Settings

```bash
python generate_video_from_image.py \
  --image video_gen_prompts/generated_image.png \
  --text video_gen_prompts/prompt+reasoning.txt \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.1-T2V-14B \
  --output_dir ./my_videos
```

### Skip Claude (Use Original Prompt)

If you want to skip the Claude API call and use the original image prompt:

```bash
python generate_video_from_image.py --skip-claude
```

## How It Works

1. **Reads Input Files**
   - Loads the generated image
   - Parses the reasoning trace and original prompt from the text file

2. **Generates Video Prompt with Claude**
   - Sends the image and reasoning to Claude 3.5 Sonnet
   - Claude analyzes the static image and creates a cinematic video prompt
   - Adds motion, camera dynamics, and temporal elements

3. **Generates Video**
   - Calls `generate.py` with the optimized prompt
   - Uses Wan 2.1 model to create the video

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | `video_gen_prompts/generated_image.png` | Path to generated image |
| `--text` | `video_gen_prompts/prompt+reasoning.txt` | Path to reasoning text file |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--task` | `t2v-14B` | Video generation task |
| `--size` | `1280*720` | Video resolution |
| `--ckpt_dir` | `./Wan2.1-T2V-14B` | Model checkpoint directory |
| `--output_dir` | (default) | Output directory for video |
| `--skip-claude` | False | Skip Claude and use original prompt |

## Input File Format

The text file should contain:

```
[Reasoning trace about how the image was created]

--- END OF REASONING ---

[Original image generation prompt]

Generating image with prompt: 
[The actual prompt used]
```

## Examples

### Example 1: Generate with all defaults
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python generate_video_from_image.py
```

### Example 2: Custom resolution and output
```bash
python generate_video_from_image.py \
  --size 1920*1080 \
  --output_dir ./high_res_videos \
  --api-key "sk-ant-..."
```

### Example 3: Different model size
```bash
python generate_video_from_image.py \
  --task t2v-1.3B \
  --ckpt_dir ./Wan2.1-T2V-1.3B
```

## Output

The script will:
1. Display the original prompt
2. Show the Claude-generated video prompt
3. Execute the video generation
4. Save the video to the output directory

Example output:
```
================================================================================
VIDEO GENERATION FROM IMAGE AND REASONING
================================================================================

📸 Image: video_gen_prompts/generated_image.png
📝 Text: video_gen_prompts/prompt+reasoning.txt
🎬 Task: t2v-14B
📐 Size: 1280*720

--------------------------------------------------------------------------------
READING REASONING AND ORIGINAL PROMPT...
--------------------------------------------------------------------------------

📖 Original prompt: Cartoon robot frantically shoving dirt under rug...

--------------------------------------------------------------------------------
GENERATING VIDEO PROMPT WITH CLAUDE...
--------------------------------------------------------------------------------

✓ Claude generated video prompt:
================================================================================
[Optimized cinematic video prompt appears here]
================================================================================

--------------------------------------------------------------------------------
GENERATING VIDEO...
--------------------------------------------------------------------------------
[Video generation progress]

✓ COMPLETE!
```

## Troubleshooting

### API Key Error
```
✗ Error: Anthropic API key required
```
**Solution**: Set the environment variable or use `--api-key` flag

### Image Not Found
```
✗ Error: Image file not found
```
**Solution**: Check the path and ensure the file exists

### Video Generation Failed
**Solution**: Ensure Wan 2.1 model is properly installed and the checkpoint directory is correct

## Tips

- Claude typically adds cinematic elements like camera movements, lighting changes, and temporal dynamics
- The script preserves the visual style and atmosphere of the original image
- For best results, ensure your reasoning trace is detailed and explains the visual elements
- You can pass additional arguments to `generate.py` by adding them after known arguments

