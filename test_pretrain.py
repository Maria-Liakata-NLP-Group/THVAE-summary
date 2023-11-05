from transformers import EncoderDecoderModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased") # initialize Bert2Bert from pre-trained checkpoints (..) –


# training (>>>) –

model.config.decoder_start_token_id = tokenizer.cls_token_id

model.config.pad_token_id = tokenizer.pad_token_id

model.config.vocab_size = model.config.decoder.vocab_size

input_ids = tokenizer

labels = tokenizer

outputs = model

loss = outputs.loss (logits)

outputs.logits

# save and load from pretrained (>>>) –

model.save_pretrained

model = EncoderDecoderModel.from_pretrained

# generation (>>>) –

generated = model.generate