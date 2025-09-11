#!/bin/bash

# Configuration
CHECKPOINT_DIR="/opt/tiger/atari_2B/checkpoints"
FRAMES_DIR="/tmp/test_frames"
PARQUET_DIR="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B/parquet_individual_v2"
FRAMES_PER_ENV=1024
MAX_EPISODES=3
RANDOMNESS=0.1
EPSILON_GREEDY=0.2  # 20% random actions for better diversity
SAVE_VIDEO=true  # Save video for each episode

echo "=== Simple Atari Pipeline ==="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Frames output: $FRAMES_DIR"
echo "Parquet output: $PARQUET_DIR"
echo ""

# Create output directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$PARQUET_DIR"

# List of all 57 environments
ENVS="
atari_alien
atari_amidar
atari_assault
atari_asterix
atari_asteroid
atari_atlantis
atari_bankheist
atari_battlezone
atari_beamrider
atari_berzerk
atari_bowling
atari_boxing
atari_breakout
atari_centipede
atari_choppercommand
atari_crazyclimber
atari_defender
atari_demonattack
atari_doubledunk
atari_enduro
atari_fishingderby
atari_freeway
atari_frostbite
atari_gopher
atari_gravitar
atari_hero
atari_icehockey
atari_jamesbond
atari_kangaroo
atari_kongfumaster
atari_krull
atari_montezuma
atari_mspacman
atari_namethisgame
atari_phoenix
atari_pitfall
atari_pong
atari_privateye
atari_qbert
atari_riverraid
atari_roadrunner
atari_robotank
atari_seaquest
atari_skiing
atari_solaris
atari_spaceinvaders
atari_stargunner
atari_surround
atari_tennis
atari_timepilot
atari_tutankham
atari_upndown
atari_venture
atari_videopinball
atari_wizardofwor
atari_yarsrevenge
atari_zaxxon
"

# Process each environment
env_num=1
for env_name in $ENVS; do
    echo "[$env_num/57] Processing $env_name..."
    
    # Remove existing files if they exist
    if [ -f "$PARQUET_DIR/${env_name}.parquet" ]; then
        rm -f "$PARQUET_DIR/${env_name}.parquet"
    fi
    
    if [ -d "$FRAMES_DIR/${env_name}" ]; then
        rm -rf "$FRAMES_DIR/${env_name}"
    fi
    
    # Run the Python script
    python3 process_individual_env.py \
        --env-name "$env_name" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --frames-dir "$FRAMES_DIR" \
        --output-dir "$PARQUET_DIR" \
        --frames-per-env $FRAMES_PER_ENV \
        --max-episodes $MAX_EPISODES \
        --randomness $RANDOMNESS \
        --epsilon-greedy $EPSILON_GREEDY \
        $([ "$SAVE_VIDEO" = "true" ] && echo "--save-video") \
        --device cpu
    
    if [ $? -eq 0 ]; then
        echo "[$env_num/57] ✓ Success: $env_name"
    else
        echo "[$env_num/57] ✗ Failed: $env_name"
    fi
    
    echo ""
    env_num=$((env_num + 1))
done

echo "=== Pipeline complete! ==="
echo "Parquet files saved to: $PARQUET_DIR"
ls -lh $PARQUET_DIR/*.parquet 2>/dev/null | wc -l | xargs echo "Total parquet files created:"