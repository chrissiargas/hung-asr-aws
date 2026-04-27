

Info = {
    'slam_asr': {
        'checkpoint_folder': None,
        'model_name': None,
        'dataset': None,
        'interleave': True,
        'speech_encoder_id': 'openai/whisper-large-v3',
        'language_model_id': 'ilsp/Llama-Krikri-8B-Instruct',
        'downsample_K': 5,
        'hidden_dim': 2048,
        'use_lora': True,
        'train_mode': 'adapter',
        'bit4': True,
        'datetime': None,
        'do_compute': True,
        'compute_wer_cer': True,
        'N_samples_for_metrics': None
    },
    'continuous_fusion': {
        'checkpoint_folder': None, # add in training
        'model_name': None,
        'dataset': None, # add in training
        'interleave': True,
        'speech_encoder_id': 'openai/whisper-large-v3',
        'language_model_id': 'ilsp/Llama-Krikri-8B-Base',
        'bit4': True,
        'datetime': None,
        'do_compute': True,
        'compute_wer_cer': True,
        'N_samples_for_metrics': None
    },
    'dual_fusion': {
        'checkpoint_folder': None, # add in training
        'model_name': None,
        'train_dataset': None, # add in training
        'interleave': True,
        'speech_encoder_id': 'openai/whisper-large-v3',
        'language_model_id': 'ilsp/Llama-Krikri-8B-Instruct',
        'bit4': True,
        'machine': 'kronos',
        'datetime': 'Mar05_11-36',
        'do_compute': True,
        'compute_wer_cer': True,
        'N_samples_for_metrics': None
    },
}