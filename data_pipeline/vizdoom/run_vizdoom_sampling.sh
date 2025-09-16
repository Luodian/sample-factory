#!/bin/bash

# Sample frames and videos from all VizDoom environments

cd /opt/tiger/sample-factory/data_pipeline

echo "=== VizDoom Sampling Script ==="
echo "Sampling frames and videos from all downloaded models"
echo ""

# doom_battle
echo "=== Sampling doom_battle ==="
python3 sample_vizdoom_simple.py \
    --env doom_battle \
    --checkpoint /tmp/vizdoom_models/doom_battle/checkpoint_p0/checkpoint_000488282_4000006144.pth \
    --output-dir /tmp/vizdoom_samples/doom_battle \
    --frames 128
echo ""

# doom_battle2
echo "=== Sampling doom_battle2 ==="
checkpoint=$(find /tmp/vizdoom_models/doom_battle2 -name "*.pth" -type f | head -1)
if [ -n "$checkpoint" ]; then
    python3 sample_vizdoom_simple.py \
        --env doom_battle2 \
        --checkpoint "$checkpoint" \
        --output-dir /tmp/vizdoom_samples/doom_battle2 \
        --frames 128
fi
echo ""

# doom_deathmatch_bots
echo "=== Sampling doom_deathmatch_bots ==="
checkpoint=$(find /tmp/vizdoom_models/doom_deathmatch_bots -name "*.pth" -type f | head -1)
if [ -n "$checkpoint" ]; then
    python3 sample_vizdoom_simple.py \
        --env doom_deathmatch_bots \
        --checkpoint "$checkpoint" \
        --output-dir /tmp/vizdoom_samples/doom_deathmatch_bots \
        --frames 128
fi
echo ""

# doom_duel_bots
echo "=== Sampling doom_duel_bots ==="
checkpoint=$(find /tmp/vizdoom_models/doom_duel_bots -name "*.pth" -type f | head -1)
if [ -n "$checkpoint" ]; then
    python3 sample_vizdoom_simple.py \
        --env doom_duel_bots \
        --checkpoint "$checkpoint" \
        --output-dir /tmp/vizdoom_samples/doom_duel_bots \
        --frames 128
fi
echo ""

# doom_duel (for doom_duel_selfplay)
echo "=== Sampling doom_duel_selfplay (as doom_duel) ==="
checkpoint=$(find /tmp/vizdoom_models/doom_duel_selfplay -name "*.pth" -type f | head -1)
if [ -n "$checkpoint" ]; then
    python3 sample_vizdoom_simple.py \
        --env doom_duel \
        --checkpoint "$checkpoint" \
        --output-dir /tmp/vizdoom_samples/doom_duel_selfplay \
        --frames 128
fi
echo ""

echo "=== Summary ==="
echo "Output directory: /tmp/vizdoom_samples"
echo ""

# Count results
if [ -d /tmp/vizdoom_samples ]; then
    for dir in /tmp/vizdoom_samples/*/; do
        if [ -d "$dir" ]; then
            env_name=$(basename "$dir")
            frame_count=$(ls "$dir"/frame_*.png 2>/dev/null | wc -l)
            video_count=$(ls "$dir"/*.mp4 2>/dev/null | wc -l)
            echo "$env_name: $frame_count frames, $video_count video(s)"
        fi
    done
else
    echo "No output directory found"
fi

echo ""
echo "Sampling complete!"