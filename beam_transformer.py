from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,LogitsProcessorList,MinLengthLogitsProcessor, \
    BeamSearchScorer, BartForConditionalGeneration, BartTokenizer
import torch


# tokenizer = AutoTokenizer.from_pretrained("t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

print(tokenizer.vocab_size)
print(tokenizer._convert_token_to_id("<s>"))
print(tokenizer._convert_token_to_id("</s>"))

encoder_input_str = list()
encoder_input_str.append("can you talk with me?")
encoder_input_str.append("can you talk with me?")
encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
tt = tokenizer._tokenize("i don't like him.")
print(tt,len(tt), 'check tokennize -=-=-=-=-=-=-')

size_list = len(encoder_input_str)

# lets run beam search using 3 beams
num_beams = 3
# define decoder start token ids
input_ids = torch.ones((num_beams * size_list, 1), device=model.device, dtype=torch.long)
print(input_ids, 'hahhahahahahah')
input_ids = input_ids * model.config.decoder_start_token_id
print(input_ids.size(), 'check later -=-=-=-=-=-=-=')

out_repeat = encoder_input_ids.repeat_interleave(num_beams, dim=0)
print(out_repeat.size(), 'check repeat -=-=-=-=-=-')

# add encoder_outputs to model keyword arguments
model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)}


encoder_output = model_kwargs['encoder_outputs']
for name in encoder_output.keys():
    print(name, 'name is what ?????/')
print(encoder_output.last_hidden_state.size())
print(encoder_output.hidden_states)

# instantiate beam scorer
beam_scorer = BeamSearchScorer(batch_size=2, max_length=model.config.max_length, num_beams=num_beams, device=model.device)

# instantiate logits processors
logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),])

outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

# tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['Wie alt bist du?']

print(tokenizer.batch_decode([14746,47, 33976, 236, 7, 213, 66, 53, 51, 860, 7, 8838 , 47 ,8 ,47 ,26 ,117 ,479,479 ,172,51 ,174 ,47,
 12196,  1102,   110,  6578,   174,    47,     7,  1095,    23,   184,     8,  4356,
   101,    99,     5 ,  856,  2420,   939, 33976,    33,    10,  6578,   479,   479,
   939 ,  216 ,   63,   142,     9,   123], skip_special_tokens=True))
