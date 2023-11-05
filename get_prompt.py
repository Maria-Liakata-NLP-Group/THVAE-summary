import torch
import re
import os
import pandas as pd

from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_llama2(folder= "/import/nlp/LLMs/llama-2/Llama-2-7b-hf"):
    tokenizer = LlamaTokenizer.from_pretrained(folder)
    model = LlamaForCausalLM.from_pretrained(folder,
                                             torch_dtype=torch.float16)  # the latter is needed so that the model fits in a single GPU
    return model, tokenizer

def load_llama(folder="/import/nlp/LLMs/llama/converted_weights/7B"):
    tokenizer = LlamaTokenizer.from_pretrained(folder)
    model = LlamaForCausalLM.from_pretrained(folder,
                                             torch_dtype=torch.float16)  # the latter is needed so that the model fits in a single GPU
    return model, tokenizer


def load_alpaca(folder="/import/nlp/LLMs/stanford_alpaca/converted_weights/"):
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForCausalLM.from_pretrained(folder,
                                                 torch_dtype=torch.float16)  # the latter is needed so that the model fits in a single GPU

    return model, tokenizer


def label_prompt(model, tokenizer, device='cuda:1', model_name='alpaca'):
    model.to(device)
    # prompts = get_prompts()
    text = get_prompts()
    # for text in prompts:
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 200)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    question = output[:output.find('<POST>')]
    example = text.split('<TEXT>')[-1]
    example_end = example.find('</TEXT>')
    example = example[:example_end]

    lbl_start_idx = len(text)
    lbl_end_idx = output[lbl_start_idx:].find('</LABEL>')
    label = output[lbl_start_idx:lbl_start_idx + lbl_end_idx]
    print('Task:\t'+question.strip())
    print('Query:\t'+example)
    print('Answer:\t'+label)
    print(output)
    print('*' * 100)


def summarize( device='cuda:1', model_name='alpaca'):

    model_id = "/import/nlp/LLMs/llama-2/Llama-2-13b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_id,  torch_dtype=torch.float16)

    model_bf16.to(device)

    # for text in prompts:

    example_english_phrase = list()
    example_english_phrase.append("i can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me.")
    example_english_phrase.append(" any one know what i should do if only someone understands me my homeschooling anywhere tomorrow.")
    example_english_phrase.append("i'm not worth it. i will never be. i'm fat, ugly, worthless, freak and invisible. ")
    example_english_phrase.append("why did i have to be born. my family will be happier if i wasn't here. ")
    example_english_phrase.append("i wish they would stop threatening me." )
    example_english_phrase.append("i don't want to do something they should leave me the fuck alone.")
    example_english_phrase.append("i'm done a dollig you can't take me everywhere i'm a human and i will cry because of what he didig i just want to be treated fair. " )
    example_english_phrase.append("i want a new start a new family a term reptiles on life number i just wanted to get better but no one thinks that. wings, labrinth! ")
    example_english_phrase.append("bff and i want to know what to do. i just don't care. in third person")


    text = f"""
    Task: Summarize the following sequence of posts in the third person.
    Context: <post>{"<post>".join(example_english_phrase)}
    Summary:
    """.strip()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model_bf16.generate(inputs["input_ids"], max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def get_prompts():
    # prompt1 = """
    # What is the main topic that concerns the user on the post?
    #     <POST><TEXT>I want to lose weight asap... I feel ugly and fat</TEXT><LABEL>weight</LABEL></POST>
    #     <POST><TEXT>I feel really nervous when surrounded by many people</TEXT><LABEL>agoraphobia</LABEL></POST>
    #     <POST><TEXT>I miss my bf so much...</TEXT><LABEL>"""
    #
    # prompt2 = """
    # Classify the presence of suicidal ideation (SI) or no (NO) in the following post. Provide the label "NA" if uncertain.
    #     <POST><TEXT>I want to lose weight asap... I feel ugly and fat</TEXT><LABEL>NO</LABEL></POST>
    #     <POST><TEXT>I feel really nervous when surrounded by many people</TEXT><LABEL>NO</LABEL></POST>
    #     <POST><TEXT>I am thinking of cutting right now</TEXT><LABEL>SI</LABEL></POST>
    #     <POST><TEXT>All I want to do now is drink till I die</TEXT><LABEL>"""
    # prompt3 = """
    # Classify the main emotion of the following post. You are free to choose any emotion. Provide the label "NA" if uncertain.
    #     <POST><TEXT>I feel glad I have finally managed to lose some weight. Yay!</TEXT><LABEL>happy</LABEL></POST>
    #     <POST><TEXT>He tells me he loves me. But when we meet, he is a different person...</TEXT><LABEL>frustrated</LABEL></POST>
    #     <POST><TEXT>The most improtant exams of my live are taking place tomorrow - can't sleep.</TEXT><LABEL>"""
    # prompt4 = """
    # Classify the main symptoms related to the user's mental health in the following post. Provide the label "NA" if none is present.
    #     <POST><TEXT>I want to lose weight asap... I feel ugly and fat</TEXT><LABEL>low self-esteem</LABEL></POST>
    #     <POST><TEXT>I feel really nervous when surrounded by many people</TEXT><LABEL>agoraphobia</LABEL></POST>
    #     <POST><TEXT>All I want to do now is drink till I die</TEXT><LABEL>self-harm ideation, alcoholism</LABEL></POST>
    #     <POST><LABEL>I miss my bf so much... i want to cut or take drugs NOW</TEXT><LABEL>"""
    # prompt5 = """
    # Classify the main relationship(s) mentioned in the following posts. Provide the label "NA" if no relationship is implied.
    #     <POST><TEXT>I like this game<\TEXT><LABEL>NA</LABEL></POST>
    #     <POST><TEXT>When I was little, my grandma used to hit me</TEXT><LABEL>family</LABEL></POST>
    #     <POST><TEXT>WMiss my bf so badly...</TEXT><LABEL>partner</LABEL></POST>
    #     <POST><TEXT>My schoolmates make fun of me every day. I am sick of them</TEXT><LABEL>"""
    # prompt6 = """
    # Choose the keywords in the following posts that can describe event and behaviour with strong emotions.
    #         <POST><TEXT>how to get rid of hunger pains without eating. I hate it that everyday I have to lie to my mom. I'm feeling guilty about it but if I tell her I'm in trouble. home alone what to do in order to keep positive and not fall through the cracks. any tips on what exercise to do in order to get a thigh gap. Well I guess my one month of no cutting is fone. I knew it the whole time I'm a failure at everything</TEXT><LABEL>get rid of hunger pains hate lie mom feeling guilty in trouble home alone keep positive cracks no cutting failure</LABEL></POST>
    #
    #         <POST><TEXT>last week snow this week rain and its April what the hell is going on this is making me depressed. My mom is making me want to see blood flowing from my skin. I already cut today I don't want more. what to do..</TEXT><LABEL>snow rain depressed mom blood flowing my skin cut</LABEL></POST>
    #
    #         <POST><TEXT>I can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me any one know what I should do. if only someone understands me. My homeschooling begins tomorrow I'm not ready for school yet. I want my cat to scratch me so I can get the relief but say I didn't cut. I actually ate and now I can't hold it down maybe I was made not to eat. Well I see my therapist tomorrow don't know if I should tell her I cut. food in my stomach feel horrible I think it should come up</TEXT><LABEL>"""
    prompt6 = """
        Choose the key phrases in the following posts that can describe event and behaviour with strong emotions.
                <POST><TEXT>how to get rid of hunger pains without eating. I hate it that everyday I have to lie to my mom. I'm feeling guilty about it but if I tell her I'm in trouble. home alone what to do in order to keep positive and not fall through the cracks. any tips on what exercise to do in order to get a thigh gap. Well I guess my one month of no cutting is fone. I knew it the whole time I'm a failure at everything</TEXT><LABEL>hunger pain, hate, lie to my mom, guilty, thigh gap, cutting, I’m a failure</LABEL></POST> 

                <POST><TEXT>last week snow this week rain and its April what the hell is going on this is making me depressed. My mom is making me want to see blood flowing from my skin. I already cut today I don't want more. what to do..</TEXT><LABEL>depressed, My mom, want to see blood, cut today, what to do</LABEL></POST> 

                <POST><TEXT>I can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me any one know what I should do. if only someone understands me. My homeschooling begins tomorrow I'm not ready for school yet. I want my cat to scratch me so I can get the relief but say I didn't cut. I actually ate and now I can't hold it down maybe I was made not to eat. Well I see my therapist tomorrow don't know if I should tell her I cut. food in my stomach feel horrible I think it should come up</TEXT><LABEL>"""

    # return [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6]
    return prompt6


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class Prompter:
    def __init__(self, is_alpaca=True, device="cuda:1"):
        if is_alpaca:
            pretrained_path = "/import/nlp/LLMs/stanford_alpaca/converted_weights/"
            print("loading alpaca")
        else:
            pretrained_path = "/import/nlp/LLMs/llama/converted_weights/7B"
            print("loading llama")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.device = device
        self.model.to(self.device)

    def run(self, prompts, max_new_tokens=200, temperature=1., top_k=None, top_p=None):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        max_length = max(len(idx) for idx in inputs.input_ids) + max_new_tokens  # update as necessary
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,

        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         <POST><TEXT>I can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing
#         me any one know what I should do. if only someone understands me. My homeschooling begins tomorrow I'm not ready for school
#         yet. I want my cat to scratch me so I can get the relief but say I didn't cut. I actually ate and now I can't hold it
#         down maybe I was made not to eat. Well I see my therapist tomorrow don't know if I should tell her I cut. food in my
#         stomach feel horrible I think it should come up.</TEXT><LABEL>if only someone understands me. I actually ate and now
#         I can't hold it down. Well I see my therapist tomorrow don't know if I should tell her I cut</LABEL></POST>


def read_timeline(model, tokenizer, device='cuda:0'):
    dir = 'Timeline_dataset/combine_addition'
    file_list = os.listdir(dir)

    prompt = """
    Choose the key phrases in the following posts that have strong emotions.
    <POST><TEXT>last week snow this week rain and its April what the he'll is going on this is making me depressed. My mom is making me want to see blood flowing from my skin. I already cut today I don't want more. what to do..</TEXT><LABEL>blood flowing from my skin. already cut today. I don't want more.</LABEL></POST>
    <POST><TEXT>I can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me any one know what I should do. if only someone understands me. My homeschooling begins tomorrow I'm not ready for school yet. I want my cat to scratch me so I can get the relief but say I didn't cut. I actually ate and now I can't hold it down maybe I was made not to eat. Well I see my therapist tomorrow don't know if I should tell her I cut. food in my stomach feel horrible I think it should come up.</TEXT><LABEL>someone understands me. I actually ate. I can't hold it down. I see my therapist tomorrow don't know if I should tell her I cut</LABEL></POST>
    """
    model.to(device)
    # prompts = get_prompts()
    # for text in prompts:

    for file in file_list:
        if file == '.DS_Store':
            continue
        from_path = os.path.join(dir, file)
        with open(from_path, 'r') as f:
            print("file name :"+ file)
            data = pd.read_csv(f, sep='\t')
            # if file == '58826_170.tsv':
            group_name = file.split('.')[0]
            group_list = list()
            col = data['review_text']
            prompt_list = list()
            for rev in col:
                post_l = rev.split()
                len_post = len(post_l)
                if len_post > 512:
                    rev = ' '.join(post_l[: 512])
                # if len(post_l[512:]) > 512:
                #     post_2 = ' '.join(post_l[512: 1024])
                # else:
                #     post_2 = ' '.join(post_l[512:])

                prompt_post = '<POST><TEXT>' + rev + '</TEXT><LABEL>'
                prompt_post = prompt + prompt_post

                inputs = tokenizer(prompt_post, return_tensors="pt")
                inputs.to(device)
                generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 200)
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                # question = output[:output.find('<POST>')]
                # example = prompt_post.split('<TEXT>')[-1]
                # example_end = example.find('</TEXT>')
                # example = example[:example_end]

                lbl_start_idx = len(prompt_post)
                lbl_end_idx = output[lbl_start_idx:].find('</LABEL>')
                # this is answer which is need to be paied attention
                label = output[lbl_start_idx:lbl_start_idx + lbl_end_idx]
                print(label)
                print('-----------')
                prompt_list.append(label)
                group_list.append(group_name)

            data['prompt'] = prompt_list
            data['group_id'] = group_list
            data.to_csv(from_path, index=False, sep='\t')

def check_timeline():
    with open('Timeline_dataset/all_label/58826_170.csv', 'r') as f:
        data = pd.read_csv(f, sep='\t')
        # if file == '58826_170.tsv':
        for index, line in data.iterrows():
            post = line['review_text']
            post_l = post.split()
            len_post = len(post_l)
            if len_post > 50:
                post_1 = ' '.join(post_l[: 50])
                print(post_1)
            if len(post_l[51 :]) > 50:
                post_2 = ' '.join(post_l[50: 100])
            else:
                post_2 = ' '.join(post_l[50:])
            print(post_2)
            print(type(post_l))
            print(len_post)

            # prompt_post = '<POST><TEXT>' + post + '</TEXT><LABEL>'
            # prompt_post = prompt + prompt_post
            #
            # inputs = tokenizer(prompt_post, return_tensors="pt")
            # inputs.to(device)
            # generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0]) + 200)
            # output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
            # question = output[:output.find('<POST>')]
            # example = prompt_post.split('<TEXT>')[-1]
            # example_end = example.find('</TEXT>')
            # example = example[:example_end]
            #
            # lbl_start_idx = len(prompt_post)
            # lbl_end_idx = output[lbl_start_idx:].find('</LABEL>')
            # label = output[lbl_start_idx:lbl_start_idx + lbl_end_idx]
            # print('Task:\t' + question.strip())
            # print('Query:\t' + example)
            # print('Answer:\t' + label)
            # print(prompt_post)




    # dir = 'Timeline_dataset/all_label'
    # file_list = os.listdir(dir)
    # count = 0
    # for file in file_list:
    #     if file == '.DS_Store':
    #         continue
    #     from_path = os.path.join(dir, file)
    #     with open(from_path, 'r') as f:
    #         data = pd.read_csv(f, sep='\t')
    #         head = data.columns
    #         if len(head) != 4:
    #             print(file)
    #             count +=1
    # print(count)

def test():
    model_id = "/import/nlp/LLMs/llama-2/Llama-2-13b-hf"
    # model_id = "meta-llama/Llama-2-13b-hf"

    # tokenizer = LlamaTokenizer.from_pretrained(model_id)
    # model_bf16 = LlamaForCausalLM.from_pretrained(model_id,
    #                                          torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        torch_dtype=torch.float16
        # device_map='auto',
    )

    device = "cuda:2"
    fp = "58826_170.tsv"

    df = pd.read_csv(fp, sep="\t")
    df = df.fillna("")
    df = df[df.content.str.strip().str.len() > 1]
    docs1 = df.content.tolist()
    doc_concat1 = " ".join(docs1)

    text = f"""
    Task: Summarize the following sequence of posts.
    Context: <post>{"<post>".join(docs1)}
    Summary:
    """.strip()

    model_bf16.to(device)

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model_bf16.generate(inputs["input_ids"], max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def get_file():
    dir = 'Timeline_dataset/combine_annotation'
    file_list = os.listdir(dir)
    for file in file_list:
        if file == '.DS_Store':
            continue
        print(file)


if __name__ == '__main__':
    # get_file()
    model, tokenizer = load_llama2()
    # label_prompt(model=model, tokenizer=tokenizer)
    read_timeline(model, tokenizer)

    # print(tokenizer.vocab_size)
    # print(model.model.config.hidden_size)

    # summarize()

    # test()



    # read_timeline(model=model, tokenizer=tokenizer)
    # check_timeline()




    # <POST><TEXT>time to get another test about 50 needles into my skin. Why me I just feel like crying.
    #             when I get home i think the blade will be my best friend. my mom needs to quit tricking me and dragging me places without telling me.
    #             I'm done with life. I'm not worth it. I will never be. I'm fat, ugly, worthless, freak and invisible. why did I have to be born. my family will be happier if I wasn't here.
    #              I wish they would stop threatening me. I'd I don't want to do something they should leave me the Fuck alone. I'm not a doll.
    #               you can't take me everywhere. I'm a human and I have my own rights and I know I'm a minor but I can do some things by myself.
    #               I don't need someone over my shoulders every minute. I need privacy. watching me all day doesn't help it just makes me worse. I done with everything.
    #               when I go to my next doctor I hope they tell me I'm very sick and that I probably will die. so maybe people will let me be me and not be someone else.
    #               I want to get better but no one in my family is making it easier for me. my mom say they will send you to your dad but I know court won't because of what he did.
    #               I just want to be treated fair. I want a new start a new family a new outlook on life. I just want to get better but no one understands that.</TEXT><LABEL>