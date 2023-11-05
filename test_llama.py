from llama_cpp import Llama
import pandas as pd


def test():
    llm = Llama(model_path='llama_model/llama-2-13b.Q4_K_M.gguf', n_ctx=2048)

    fp = "58826_170.tsv"

    df = pd.read_csv(fp, sep="\t")
    df = df.fillna("")
    df = df[df.content.str.strip().str.len() > 1]
    docs1 = df.content.tolist()
    doc_concat1 = " ".join(docs1)

    h_vae = list()
    h_vae.append("i can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me")
    h_vae.append("any one know what i should do if only someone understands me my homeschooling anywhere tomorrow")
    h_vae.append("i'm not worth it. i will never be. i'm fat, ugly, worthless, freak and invisible. why did i have to be born. my family will be happier if i wasn't here.")
    h_vae.append("i wish they would stop threatening me  i don't want to do something they should leave me the fuck alone.")
    h_vae.append("i'm done a dollig you can't take me everywhere i'm a human and i will cry because of what he didig i just want to be treated fair.")
    h_vae.append("i want a new start a new family a term reptiles on life number i just wanted to get better but no one thinks that.")
    h_vae.append("wings, labrinth! bff and i want to know what to do. i just don't care.")

    test_text = "how to get rid of hunger pains without eating  I hate it that everyday I have to lie to my mom . I'm feeling guilty about it but if I tell her I'm in trouble.  home alone what to do in order to keep positive and not fall through the cracks.  any tips on what exercise to do in order to get a thigh gap.  Well I guess my one month of no cutting is fone . I knew it the whole time I'm a failure at everything"
    test_text2 = "last week snow this week rain and its April what the he'll is going on this is making me depressed. My mom is making me want to see blood flowing from my skin. I already cut today I don't want more. what to do.."
    test_text3 = "time to get another test about 50 needles into my skin Why me I just feel like crying . when I get home i think the blade will be my best friend . my mom needs to quit tricking me and dragging me places without telling me . I'm done with life . I'm not worth it . I will never be . I'm fat , ugly , worthless , freak and invisible . why did I have to be born . my family Will be happier if I wasn't here . I wish they would stop threatening me ."
    test_text4 = "I can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me any one know what I should do.  if only someone understands me.  My homeschooling begins tomorrow I'm not ready for school yet  I want my cat to scratch me so I can get the relief but say I didn't cut  I actually ate and now I can't hold it down maybe I was made not to eat  Well I see my therapist tomorrow don't know if I should tell her I cut  food in my stomach feel horrible I think it should come up"
    # Context: {".".join(docs1)}
    text = f"""
        Task: Choose the key phrases in the following posts with strong emotion.
        Context: last week snow this week rain and its April what the he'll is going on this is making me depressed. My mom is making me want to see blood flowing from my skin. I already cut today I don't want more. what to do..
        key phrases: blood flowing from my skin. already cut today. I don't want more.
        Context: I can't hold down any food so my mom says take a anti acid but it doesn't help and now my throat is killing me any one know what I should do. if only someone understands me. My homeschooling begins tomorrow I'm not ready for school yet. I want my cat to scratch me so I can get the relief but say I didn't cut. I actually ate and now I can't hold it down maybe I was made not to eat. Well I see my therapist tomorrow don't know if I should tell her I cut. food in my stomach feel horrible I think it should come up.
        key phrases: someone understands me. I actually ate. I can't hold it down. I see my therapist tomorrow don't know if I should tell her I cut.
        Context: {test_text3}
        key phrases:
        """.strip()
    # temperature=0.8, top_p=0.95, top_k=40, repeat_penalty=1,temperature=1.3,  # Context: {test_text}
    #         # key phrases
    # , temperature=1.2,  top_k=5, echo=True
    output = llm(prompt=text, temperature=1.2,  echo=True)
    choice = output['choices']
    key_phrases = choice[0]['text']
    key_phrase = key_phrases.split('key phrases:')[-1]
    key = key_phrases.split(' Context:')[0]
    # kk = key_phrases.split("\""" ")
    print('text ================')
    print(key)
    print('KKKKKKKKKKKKKK',key_phrase, '===============')
    print(output)

if __name__ == '__main__':

    test()
