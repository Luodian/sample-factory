# Atari Training Data Pipeline

## Overview
Complete pipeline for sampling frames from Atari environments and converting them to training data format with interleaved images and text.

## Pipeline Steps

### 1. Sample Frames from Environments
```bash
# Sample frames with episode organization
python3 sample_all_envs.py \
    --frames-per-env 60 \
    --max-episodes 3 \
    --randomness 0.2 \
    --parallel 8 \
    --output-dir sampled_frames_organized
```

This creates:
```
sampled_frames_organized/
└── atari_alien/
    ├── episode_000_rand0.2/
    │   ├── frame_000000.png
    │   ├── action_000000.txt (contains: "8: DOWNRIGHT")
    │   └── ...
    └── episode_001_rand0.2/
        └── ...
```

### 2. Convert to Training Data Format
```bash
# Create interleaved format (episodes as sequences)
python3 convert_to_training_data.py \
    --input-dir sampled_frames_organized \
    --output-file atari_training_data.parquet \
    --format interleaved

# Or create simple format (one row per frame)
python3 convert_to_training_data.py \
    --input-dir sampled_frames_organized \
    --output-file atari_training_data_simple.parquet \
    --format simple
```

## Output Formats

### Interleaved Format (Training-Ready)
Each row represents a complete episode with:
- `inputs`: List of dictionaries with interleaved text and image_gen entries
  - Text entries: `{"type": "text", "has_loss": 0/1, "text": "..."}`
  - Image entries: `{"type": "image_gen", "has_loss": 1, "image_index": N}`
- `images`: List of PNG image bytes
- `images_front`: First frame as front image
- `environment`: Environment name
- `episode`: Episode identifier  
- `num_frames`: Number of frames in episode
- `randomness`: Randomness level used

Example input sequence:
```python
[
  {"type": "text", "has_loss": 0, "text": "Atari Alien Environment (randomness=0.2)"},
  {"type": "text", "has_loss": 1, "text": "Episode episode_000_rand0.2. Starting game with 11 frames."},
  {"type": "image_gen", "has_loss": 1, "image_index": 0},
  {"type": "text", "has_loss": 0, "text": "downright"},
  {"type": "image_gen", "has_loss": 1, "image_index": 1},
  ...
]
```

### Simple Format
Each row represents a single frame:
- `environment`: Environment name
- `episode`: Episode identifier  
- `frame_number`: Frame index
- `action`: Cleaned action string (digit prefix removed)
- `action_raw`: Original action with digit
- `image`: Image bytes

## Key Features

### Action Cleaning
Actions are automatically cleaned:
- Original: `"8: DOWNRIGHT"`
- Cleaned: `"DOWNRIGHT"`

### Episode Organization
- Each episode saved in separate folder
- Randomness level included in folder name
- Maintains temporal sequence

### Flexible Sampling
- Control frames per episode
- Set number of episodes
- Adjust randomness (0.0 = deterministic, 1.0 = fully stochastic)

## Full Pipeline Example

```bash
# 1. Sample from all Atari environments
python3 sample_all_envs.py \
    --frames-per-env 100 \
    --max-episodes 5 \
    --randomness 0.2 \
    --parallel 8

# 2. Convert to training data
python3 convert_to_training_data.py \
    --input-dir sampled_frames_organized \
    --output-file atari_full_dataset.parquet \
    --format interleaved

# 3. Verify the output
python3 -c "
import pandas as pd
df = pd.read_parquet('atari_full_dataset.parquet')
print(f'Dataset shape: {df.shape}')
print(f'Environments: {df[\"environment\"].unique()}')
print(f'Total episodes: {len(df)}')
"
```

## Usage Tips

1. **Memory Management**: For large datasets, process environments in batches
2. **Action Diversity**: Use randomness 0.2-0.5 for good action variety
3. **Episode Length**: Adjust frames-per-env based on typical episode length
4. **Parallel Processing**: Use multiple workers for faster sampling

## Reading the Data

```python
import pandas as pd
from PIL import Image
import io

# Load dataset
df = pd.read_parquet('atari_training_data.parquet')

# Access first episode
episode = df.iloc[0]
print(f"Episode: {episode['episode']}")
print(f"Randomness: {episode['randomness']}")
print(f"Number of frames: {episode['num_frames']}")

# Process inputs
for i, inp in enumerate(episode['inputs'][:5]):
    if inp['type'] == 'text':
        print(f"  Text: {inp['text']}")
    else:
        print(f"  Image at index {inp['image_index']}")

# Load first image
img_bytes = episode['images'][0]
img = Image.open(io.BytesIO(img_bytes))
img.show()
```

## Notes
- Images are stored as PNG bytes for efficiency
- Actions are human-readable (UPRIGHT, FIRE, etc.)
- Supports both sequential and frame-by-frame training approaches