# Frame-by-Frame Saving with Human-Readable Actions

## Summary
The frame saving feature has been enhanced to save actions with human-readable names instead of just integer values.

## Key Updates

### 1. Human-Readable Action Names
Actions are now saved with their meanings for Atari games:
- **Before**: `[[11]]` (just the integer)
- **After**: `11: RIGHTFIRE` (index + meaning)

### 2. Atari Action Meanings
The system automatically detects and uses action meanings from the Atari environment:
- NOOP, FIRE, UP, RIGHT, LEFT, DOWN
- UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT
- UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE
- UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE

### 3. Usage Example
```bash
python3 -m sf_examples.atari.enjoy_atari \
    --env=atari_alien \
    --experiment=edbeeching_atari_2B_atari_alien_1111 \
    --train_dir=/opt/tiger/sample-factory/checkpoints \
    --device=cpu \
    --save_frames \
    --frames_dir=frames_output \
    --video_frames=500 \
    --no_render
```

### 4. Output Format
- Frames: `frame_000000.png`, `frame_000001.png`, ...
- Actions: `action_000000.txt`, `action_000001.txt`, ...
- Action content example: `8: DOWNRIGHT`

## Implementation Details
- Modified `/opt/tiger/sample-factory/sample_factory/enjoy.py`:
  - `save_frame_to_disk()` function now accepts `action_meanings` parameter
  - Automatically retrieves action meanings from `env.unwrapped.get_action_meanings()`
  - Maps action indices to their string representations

## Benefits
- **Easier Analysis**: Researchers can understand agent behavior without memorizing action codes
- **Better Debugging**: Quickly identify what actions the agent is taking
- **Documentation**: Action files serve as self-documenting behavior logs