#!/bin/bash

# Simplified pipeline for sampling frames using best checkpoints
# Each environment uses its best performing checkpoint (typically from _3333 suffix)

# Configuration
CHECKPOINT_DIR="/opt/tiger/sample-factory/checkpoints"
OUTPUT_BASE="/mnt/bn/seed-aws-va/brianli/prod/contents/atari_2B"
FRAMES_DIR="${OUTPUT_BASE}/sampled_frames_best"
VIDEO_DIR="${OUTPUT_BASE}/sampled_videos_best"

# Sampling parameters
MAX_EPISODES=${MAX_EPISODES:-10}
MAX_FRAMES=${MAX_FRAMES:-2400}  # 240 frames * 10 episodes
FPS=${FPS:-30}
DEVICE=${DEVICE:-"cpu"}

# Parallel configuration
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

echo "=== Atari Best Checkpoint Sampling Pipeline ==="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Frames output: $FRAMES_DIR"
echo "Videos output: $VIDEO_DIR"
echo "Max episodes: $MAX_EPISODES"
echo "Max frames: $MAX_FRAMES"
echo "Parallel jobs: $PARALLEL_JOBS"
echo ""

# Create output directories
mkdir -p "$FRAMES_DIR"
mkdir -p "$VIDEO_DIR"

# Define environments and their best checkpoints
# Format: env_name:suffix (suffix indicates which model version to use)
# You can comment out any environments you don't want to process
declare -a ENV_CONFIGS=(
    "atari_alien:3333"
    "atari_amidar:3333"
    "atari_assault:3333"
    "atari_asterix:3333"
    "atari_asteroid:3333"
    "atari_atlantis:3333"
    "atari_bankheist:3333"
    "atari_battlezone:3333"
    "atari_beamrider:3333"
    "atari_berzerk:3333"
    "atari_bowling:3333"
    "atari_boxing:3333"
    "atari_breakout:3333"
    "atari_centipede:3333"
    "atari_choppercommand:3333"
    "atari_crazyclimber:3333"
    "atari_defender:3333"
    "atari_demonattack:3333"
    "atari_doubledunk:3333"
    "atari_enduro:3333"
    "atari_fishingderby:3333"
    "atari_freeway:3333"
    "atari_frostbite:3333"
    "atari_gopher:3333"
    "atari_gravitar:3333"
    "atari_hero:3333"
    "atari_icehockey:3333"
    "atari_jamesbond:3333"
    "atari_kangaroo:3333"
    "atari_kongfumaster:3333"
    "atari_krull:3333"
    "atari_montezuma:3333"
    "atari_mspacman:3333"
    # "atari_namethisgame:3333"
    # "atari_phoenix:3333"
    # "atari_pitfall:3333"
    # "atari_pong:3333"
    # "atari_privateye:3333"
    # "atari_qbert:3333"
    # "atari_riverraid:3333"
    # "atari_roadrunner:3333"
    # "atari_robotank:3333"
    # "atari_seaquest:3333"
    # "atari_skiing:3333"
    # "atari_solaris:3333"
    # "atari_spaceinvaders:3333"
    # "atari_stargunner:3333"
    # "atari_surround:3333"
    # "atari_tennis:3333"
    # "atari_timepilot:3333"
    # "atari_tutankham:3333"
    # "atari_upndown:3333"
    # "atari_venture:3333"
    # "atari_videopinball:3333"
    # "atari_wizardofwor:3333"
    # "atari_yarsrevenge:3333"
    # "atari_zaxxon:3333"
)

# Function to process a single environment
process_environment() {
    local config=$1
    IFS=':' read -r env_name suffix <<< "$config"

    echo "[$(date '+%H:%M:%S')] Processing $env_name with checkpoint $suffix"

    local experiment="edbeeching_atari_2B_${env_name}_${suffix}"
    local env_frames_dir="${FRAMES_DIR}/${env_name}"
    local env_video_path="${VIDEO_DIR}/${env_name}.mp4"

    # Check if checkpoint exists
    if [ ! -d "${CHECKPOINT_DIR}/${experiment}" ]; then
        echo "[$(date '+%H:%M:%S')] WARNING: Checkpoint not found for $env_name ($suffix), skipping..."
        return 1
    fi

    # Clean up any existing frames for this env
    rm -rf "$env_frames_dir"
    mkdir -p "$env_frames_dir"

    # Sample frames using the best checkpoint (saves each episode to a separate folder)
    echo "[$(date '+%H:%M:%S')] Sampling frames for $env_name..."
    python /opt/tiger/sample-factory/data_pipeline/enjoy_atari_episode_folders.py \
        --env "$env_name" \
        --experiment "$experiment" \
        --train_dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --save_frames \
        --frames_dir "$env_frames_dir" \
        --max_num_frames "$MAX_FRAMES" \
        --max_num_episodes "$MAX_EPISODES" \
        --no_render \
        --load_checkpoint_kind best \
        2>&1 | sed "s/^/[$env_name] /"

    if [ $? -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: Failed to sample frames for $env_name"
        return 1
    fi

    # Count frames and episodes
    local episode_count=$(find "$env_frames_dir" -type d -name "episode_*" 2>/dev/null | wc -l)
    local frame_count=$(find "$env_frames_dir" -name "*.png" 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] Sampled $frame_count frames across $episode_count episodes for $env_name"

    # Create video from frames
    if [ "$frame_count" -gt 0 ]; then
        echo "[$(date '+%H:%M:%S')] Creating video for $env_name..."

        # Create a combined video from all episode frames
        # First, create a file list with all frames in order
        local frame_list_file="${env_frames_dir}/framelist.txt"
        find "${env_frames_dir}" -name "*.png" | sort | while read frame; do
            echo "file '$frame'" >> "$frame_list_file"
        done

        # Use ffmpeg to create video from the frame list
        ffmpeg -y -f concat -safe 0 -i "$frame_list_file" \
            -framerate "$FPS" \
            -c:v libx264 -pix_fmt yuv420p -crf 23 \
            "$env_video_path" \
            2>&1 | grep -v "frame=" | sed "s/^/[$env_name] /"

        # Clean up the frame list file
        rm -f "$frame_list_file"

        if [ -f "$env_video_path" ]; then
            local video_size=$(du -h "$env_video_path" | cut -f1)
            echo "[$(date '+%H:%M:%S')] Created video: ${env_name}.mp4 ($video_size)"
        else
            echo "[$(date '+%H:%M:%S')] ERROR: Failed to create video for $env_name"
        fi
    fi

    echo "[$(date '+%H:%M:%S')] Completed $env_name"
    return 0
}

# Export function and variables for parallel execution
export -f process_environment
export CHECKPOINT_DIR FRAMES_DIR VIDEO_DIR MAX_EPISODES MAX_FRAMES FPS DEVICE

# Start processing
START_TIME=$(date +%s)

echo "Starting parallel processing of ${#ENV_CONFIGS[@]} environments..."
echo "=========================================="

# Process environments in parallel
printf '%s\n' "${ENV_CONFIGS[@]}" | \
    xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'process_environment "$@"' _ {}

# Summary statistics
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Pipeline completed in $((DURATION / 60))m $((DURATION % 60))s"
echo ""
echo "Output locations:"
echo "  Frames: $FRAMES_DIR"
echo "  Videos: $VIDEO_DIR"
echo ""

# Count results
if [ -d "$FRAMES_DIR" ]; then
    echo "Frames and episodes per environment:"
    for env_dir in "$FRAMES_DIR"/*; do
        if [ -d "$env_dir" ]; then
            env_name=$(basename "$env_dir")
            episode_count=$(find "$env_dir" -type d -name "episode_*" 2>/dev/null | wc -l)
            frame_count=$(find "$env_dir" -name "*.png" 2>/dev/null | wc -l)
            echo "  $env_name: $episode_count episodes, $frame_count frames"
        fi
    done
fi

echo ""
if [ -d "$VIDEO_DIR" ]; then
    video_count=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    echo "Total videos created: $video_count"
    ls -lh "$VIDEO_DIR"/*.mp4 2>/dev/null | head -10
fi