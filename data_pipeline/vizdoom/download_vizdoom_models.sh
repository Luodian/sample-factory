#!/bin/bash

# Download VizDoom models from HuggingFace
# Downloads all 5 pre-trained models directly

# Configuration
MODEL_DIR="${VIZDOOM_MODEL_DIR:-/tmp/vizdoom_models}"

echo "=== VizDoom Model Downloader ==="
echo "Downloading all VizDoom models to: $MODEL_DIR"
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    uv pip install -q huggingface-hub
fi

echo "Starting downloads..."
echo ""

# Download doom_battle
echo "1/5: Downloading doom_battle..."
huggingface-cli download andrewzhang505/sample-factory-2-doom-battle \
    --local-dir "$MODEL_DIR/doom_battle" \
    --quiet 2>/dev/null || echo "  Warning: doom_battle download had issues"

# Find and rename checkpoint
if [ -f "$MODEL_DIR/doom_battle/checkpoint_p0/checkpoint_000000978_4001792.pth" ]; then
    mv "$MODEL_DIR/doom_battle/checkpoint_p0/checkpoint_000000978_4001792.pth" \
       "$MODEL_DIR/doom_battle/checkpoint.pth"
fi
echo "  ✓ doom_battle downloaded"
echo ""

# Download doom_battle2
echo "2/5: Downloading doom_battle2..."
huggingface-cli download andrewzhang505/sample-factory-2-doom-battle2 \
    --local-dir "$MODEL_DIR/doom_battle2" \
    --quiet 2>/dev/null || echo "  Warning: doom_battle2 download had issues"

# Find and rename checkpoint
for ckpt in "$MODEL_DIR/doom_battle2"/*/*.pth "$MODEL_DIR/doom_battle2"/*.pth; do
    if [ -f "$ckpt" ]; then
        mv "$ckpt" "$MODEL_DIR/doom_battle2/checkpoint.pth" 2>/dev/null
        break
    fi
done
echo "  ✓ doom_battle2 downloaded"
echo ""

# Download doom_deathmatch_bots
echo "3/5: Downloading doom_deathmatch_bots..."
huggingface-cli download andrewzhang505/doom_deathmatch_bots \
    --local-dir "$MODEL_DIR/doom_deathmatch_bots" \
    --quiet 2>/dev/null || echo "  Warning: doom_deathmatch_bots download had issues"

# Find and rename checkpoint
for ckpt in "$MODEL_DIR/doom_deathmatch_bots"/*/*.pth "$MODEL_DIR/doom_deathmatch_bots"/*.pth; do
    if [ -f "$ckpt" ]; then
        mv "$ckpt" "$MODEL_DIR/doom_deathmatch_bots/checkpoint.pth" 2>/dev/null
        break
    fi
done
echo "  ✓ doom_deathmatch_bots downloaded"
echo ""

# Download doom_duel_bots
echo "4/5: Downloading doom_duel_bots..."
huggingface-cli download andrewzhang505/doom_duel_bots_pbt \
    --local-dir "$MODEL_DIR/doom_duel_bots" \
    --quiet 2>/dev/null || echo "  Warning: doom_duel_bots download had issues"

# Find and rename checkpoint
for ckpt in "$MODEL_DIR/doom_duel_bots"/*/*.pth "$MODEL_DIR/doom_duel_bots"/*.pth; do
    if [ -f "$ckpt" ]; then
        mv "$ckpt" "$MODEL_DIR/doom_duel_bots/checkpoint.pth" 2>/dev/null
        break
    fi
done
echo "  ✓ doom_duel_bots downloaded"
echo ""

# Download doom_duel_selfplay
echo "5/5: Downloading doom_duel_selfplay..."
huggingface-cli download andrewzhang505/doom-duel-selfplay \
    --local-dir "$MODEL_DIR/doom_duel_selfplay" \
    --quiet 2>/dev/null || echo "  Warning: doom_duel_selfplay download had issues"

# Find and rename checkpoint
for ckpt in "$MODEL_DIR/doom_duel_selfplay"/*/*.pth "$MODEL_DIR/doom_duel_selfplay"/*.pth; do
    if [ -f "$ckpt" ]; then
        mv "$ckpt" "$MODEL_DIR/doom_duel_selfplay/checkpoint.pth" 2>/dev/null
        break
    fi
done
echo "  ✓ doom_duel_selfplay downloaded"
echo ""

echo "=== Download Complete ==="
echo ""
echo "Checking downloaded models..."

# Check what was actually downloaded
count=0
for env in doom_battle doom_battle2 doom_deathmatch_bots doom_duel_bots doom_duel_selfplay; do
    if [ -d "$MODEL_DIR/$env" ]; then
        # Look for any .pth file
        checkpoint=$(find "$MODEL_DIR/$env" -name "*.pth" -type f 2>/dev/null | head -1)
        if [ -n "$checkpoint" ]; then
            size_mb=$(du -m "$checkpoint" | cut -f1)
            echo "  ✓ $env: Found checkpoint (${size_mb}MB)"
            ((count++))
        else
            echo "  ✗ $env: Directory exists but no checkpoint found"
        fi
    else
        echo "  ✗ $env: Not downloaded"
    fi
done

echo ""
echo "Successfully downloaded: $count/5 models"
echo ""
echo "Models are ready for use with the sampling pipeline."
echo "To sample from these models, run:"
echo "  ./vizdoom_pipeline.sh"

exit 0