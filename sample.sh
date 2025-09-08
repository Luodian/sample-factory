python3 -m sf_examples.atari.enjoy_atari \
    --env=atari_alien \
    --experiment=atari_2B_atari_alien_1111 \
    --train_dir=./checkpoints \
    --device=cpu \
    --save_video \
    --video_frames=500 \
    --max_num_episodes=2 \
    --no_render