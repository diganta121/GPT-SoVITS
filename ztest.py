from gpt_sovits import TTS, TTS_Config

soviets_configs = {
    "default": {
        "device": "cuda",  #  ["cpu", "cuda"]
        "is_half": True,  #  Set 'False' if you will use cpu
        "t2s_weights_path": "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "vits_weights_path": "pretrained_models/s2G488k.pth",
        "cnhuhbert_base_path": "pretrained_models/chinese-hubert-base",
        "bert_base_path": "pretrained_models/chinese-roberta-wwm-ext-large",
    }
}
# load model
tts_config = TTS_Config(soviets_configs)
tts_pipeline = TTS(tts_config)

params = {
    "text": "hello i am an ai",  # str.(required) text to be synthesized
    "text_lang": "en",  # str.(required) language of the text to be synthesized [, "en", "zh", "ja", "all_zh", "all_ja"]
    "ref_audio_path": "",  # str.(required) reference audio path
    "prompt_text": "",  # str.(optional) prompt text for the reference audio
    "prompt_lang": "",  # str.(required) language of the prompt text for the reference audio
    "top_k": 5,  # int. top k sampling
    "top_p": 1,  # float. top p sampling
    "temperature": 1,  # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see gpt_sovits_python\TTS_infer_pack\text_segmentation_method.py for details.
    "batch_size": 1,  # int. batch size for inference
    "batch_threshold": 0.75,  # float. threshold for batch splitting.
    "split_bucket": True,  # bool. whether to split the batch into multiple buckets.
    "speed_factor": 1.0,  # float. control the speed of the synthesized audio.
    "fragment_interval": 0.3,  # float. to control the interval of the audio fragment.
    "seed": -1,  # int. random seed for reproducibility.
    "media_type": "wav",  # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": False,  # bool. whether to return a streaming response.
    "parallel_infer": True,  # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35,  # float.(optional) repetition penalty for T2S model.
}

# inference
tts_generator = tts_pipeline.run(params)
sr, audio_data = next(tts_generator)

# save
import scipy

scipy.io.wavfile.write("out_from_text.wav", rate=sr, data=audio_data)
