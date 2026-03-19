 - whisper-large-v3-turbo-gl-v1.0 now uses a dedicated whisper_seq2seq path with AutoModelForSpeechSeq2Seq + AutoProcessor.
 - w2v-bert-2.0-gl now uses a dedicated wav2vec2_bert_ctc path with AutoModelForCTC + AutoProcessor.
 - phi-4-multimodal-instruct-gl-v1.0 still uses the model-specific multimodal generation path, but now loads through a sanitized local alias path so trust_remote_code won’t choke on the original directory name.

 whisper-large-v3-turbo-gl-v1.0, w2v-bert-2.0-gl, phi-4-multimodal-instruct-gl-v1.0, 
