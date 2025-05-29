import transformers
from transformers import BartTokenizer, BartModel
from operator import itemgetter

from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import transformers.generation.utils as util






def test():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    vocab = tokenizer.get_vocab()
    print(type(vocab), len(vocab), 'word vocab 00-0-0-0-0-')
    # satrt is <s> end is </s>
    input_list = list()
    input_list.append("time to get another test about 50 needles into my skin. ")
    # input_list.append('who are you.')
    # input_list.append('my dog is cute.')
    keys = ['input_ids', 'attention_mask']
    output = tokenizer(input_list, padding='longest', truncation=True, return_tensors='pt')
    a,b = itemgetter(*keys)(output)

    print(a.size(), b)
    print(output)

    pre = tokenizer.decode(a.squeeze(0))
    print(pre)

def generate_summary():


    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
    example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
    batch = tok(example_english_phrase, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"])

    assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
        "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
    ]
    print(tok.batch_decode(generated_ids, skip_special_tokens=True))


    # model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    #
    # ARTICLE_TO_SUMMARIZE = (
    #     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    #     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    #     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    # )
    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
    #
    # # Generate Summary
    # summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    # summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(summary, 'summary -=-=-=-=')

def check_parameter():
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base',)
    for name, param in model.named_parameters():

        print(name ,'    ', param.size())


if __name__ == '__main__':
    # generate_summary()
    # test()
    check_parameter()

