import os
version = "v1.0"  # JavisDiT-v1.0 (JavisDiT++)

WEIGHT_ROOT = os.environ.get("WEIGHT_ROOT", "../../weights")

# Data settings
num_frames = 65  # 4s
image_size = (240, 432)  # height240, width426 = 240p
video_fps = 16
frame_interval = 1
direct_load_video_clip = True
pre_tokenize = True

# Save settings
audio_fps = 16000
save_fps = 16
multi_resolution = "OpenSora"
condition_frame_length = 5  # used for video extension conditioning
align = 5  # TODO: unknown mechanism, maybe for conditional frame alignment?

# Model settings
lora_dir = "lora"
model = dict(
    type="Wan2_1_T2V_1_3B",
    weight_init_from=f"{WEIGHT_ROOT}/JavisVerse/JavisDiT-v1.0-jav",
    model_type='t2av',
    patch_size=(1, 2, 2),
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    num_heads=12,
    num_layers=30,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    audio_patch_size=(2, 2),
    audio_in_dim=8,
    audio_out_dim=8,
    audio_special_token=False,
    train_audio_specific_blocks=False,  # do not train alone
    dual_ffn=True,
    init_from_video_branch=False,
    class_drop_prob=0.0,
    audio_pe_type='interleave_window_offset',
    init_to_device=False,
)
vae = dict(
    type="Wan2_1_T2V_1_3B_VAE",
    from_pretrained=f"{WEIGHT_ROOT}/pretrained/dit/Wan2.1-T2V-1.3B",
    vae_checkpoint='Wan2.1_VAE.pth',
    vae_stride=(4, 8, 8),
    init_to_device=False,
)
audio_vae = dict(
    type="AudioLDM2",
    from_pretrained=f"{WEIGHT_ROOT}/pretrained/dit/audioldm2",
    init_to_device=False,
)
text_encoder_output_dim = 4096
text_encoder_model_max_length = 512
text_encoder = dict(
    type="Wan2_1_T2V_1_3B_t5_umt5",
    from_pretrained=f"{WEIGHT_ROOT}/pretrained/dit/Wan2.1-T2V-1.3B",
    t5_checkpoint='models_t5_umt5-xxl-enc-bf16.pth',
    t5_tokenizer='google/umt5-xxl',
    text_len=text_encoder_model_max_length,
    init_to_device=False,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
    num_sampling_steps=50,
    transform_scale=5.0,
)

aes = None   # aesthetic score
flow = None  # motion score
neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，低音质，差音质，最差音质，噪音，失真的，破音，削波失真，数字瑕疵，声音故障，不自然的，刺耳的，尖锐的，底噪，过多混响，过多回声，突兀的剪辑，不自然的淡出，录音质量差，业余录音'

# audio settings
sampling_rate = 16000
mel_bins = 64
audio_cfg = {
    "preprocessing": {
        "audio": {
            "sampling_rate": sampling_rate,
            "max_wav_value": 32768.0,
            "duration": 10.24,
            "scale_factor": 8 # pad 1 token at most.
        },
        "stft": {
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 1024,
        },
        "mel": {
            "n_mel_channels": mel_bins,
            "mel_fmin": 0,
            "mel_fmax": 8000,
        }
    },
    "augmentation": {
        "mixup": 0.0,
    }
}