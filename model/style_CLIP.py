# from transformers import BertTokenizer, BertModel
import os
import pickle
from pathlib import Path

from transformers import AutoModel,AutoTokenizer
import torch

import clip
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False



# 输入文本
text=[]
text.append("Breakdance, also known as B-boying or breaking, is an energetic and acrobatic style of dance that originated in the streets of New York City. It combines intricate footwork, power moves, freezes, and energetic spins on the floor. Breakdancers showcase their strength, agility, and creativity through explosive movements and impressive stunts. This dynamic dance style has become a global phenomenon, inspiring dancers worldwide to push their physical limits and express themselves through the art of breaking.")
text.append("House dance is a vibrant and soulful style that originated in the underground clubs of Chicago and New York City. It combines elements of disco, funk, and hip-hop, creating a unique fusion of footwork, fluid movements, and intricate rhythms. House dancers often freestyle to electronic music, allowing their bodies to flow with the infectious beats. With its emphasis on individual expression and improvisation, house dance embodies the spirit of freedom and community within the dance culture.")
text.append("Jazz ballet, also known as contemporary jazz, is a fusion of classical ballet technique and the expressive nature of jazz dance. It combines the grace and precision of ballet with the rhythmic and syncopated movements of jazz. Jazz ballet dancers exhibit a strong sense of musicality, emphasizing body isolations, turns, jumps, and extensions. This versatile dance style allows for both structured choreography and personal interpretation, making it a popular choice for dancers seeking a balance between technique and creative expression.")
text.append("Street Jazz is a dance form that combines elements of street dance and jazz dance. It embodies the characteristics of street culture and urban vibes, emphasizing freedom, individuality, and a sense of rhythm. The dance steps of Street Jazz include fluid body movements, sharp footwork, and dynamic spins. Dancers express themselves freely on stage, showcasing their personality and sense of style through dance. Street Jazz is suitable for individuals who enjoy dancing, have a passion for street culture, and seek to express their individuality.")
text.append("Krump is an expressive and aggressive street dance style that originated in the neighborhoods of Los Angeles. It is characterized by its intense and energetic movements, including chest pops, stomps, arm swings, and exaggerated facial expressions. Krump dancers use their bodies as a form of personal expression, channeling their emotions and energy into powerful and raw performances. This urban dance form serves as an outlet for self-expression and has evolved into a community-driven movement promoting positivity and self-empowerment.")
text.append("Los Angeles-style hip-hop dance, also known as LA-style, is a fusion of various street dance styles that emerged from the urban communities of Los Angeles. It combines elements of popping, locking, breaking, and freestyle, creating a dynamic and diverse dance form. LA-style dancers showcase their unique styles and personalities through intricate body isolations, fluid movements, and sharp hits. This dance style embodies the vibrant and eclectic culture of Los Angeles and has influenced the global hip-hop dance scene.")
text.append("Locking is a funk-based dance style that originated in the 1970s. It is characterized by its distinctive moves known as locks, which involve freezing in certain positions and then quickly transitioning to the next. Locking combines funky footwork, energetic arm gestures, and exaggerated body movements, creating a playful and entertaining dance form. Lockers often incorporate humor and showmanship into their performances, making it a popular style for entertainment purposes.")
text.append("Middle Hip-Hop is a dance form that combines various elements of street dance, including popping and locking, with the rhythms and beats of hip-hop music. It requires a great deal of rhythm, coordination, and control. Middle Hip-Hop features energetic and dynamic movements, such as intricate footwork, isolations, and body waves. Dancers express themselves through their movements, delivering a powerful and engaging performance that captivates audiences. Middle Hip-Hop provides a great way for individuals to enhance their overall fitness, build confidence, and express their creativity.")
text.append("Popular dance encompasses a wide range of contemporary dance styles that are widely enjoyed and accessible to the general public. It includes various genres such as commercial dance, music video choreography, and social dances like line dancing or party dances. Popular dancers often adapt and fuse different dance styles, creating visually appealing routines that resonate with a broad audience. This dance category reflects the ever-evolving trends and influences in popular culture.")
text.append("Waacking, also known as punking or whacking, is a dance style that originated in the LGBTQ+ clubs of 1970s Los Angeles. It is characterized by its fluid arm movements, dramatic poses, and expressive storytelling through dance. Waacking requires precision, musicality, and personality to convey emotions and narratives. Dancers often use props like fans or scarves to enhance their performances. This dance form celebrates individuality, self-expression, and the liberation of one's true self.")

style=[]
style.append("mBR")
style.append("mHO")
style.append("mJB")
style.append("mJS")
style.append("mKR")
style.append("mLH")
style.append("mLO")
style.append("mMH")
style.append("mPO")
style.append("mWA")

features ={}

for i in range(len(text)):
    t = clip.tokenize([text[i]], truncate=True).cuda()
    feature = clip_model.encode_text(t).cpu().float().reshape(-1)

    # print(features)
    print(feature.shape)  # (1,768)
    features.update({style[i]:feature})

    save_path = "../data/style_clip"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    outname = style[i] +'.pkl'
    pickle.dump(
        {
            style[i]:feature
        },
        open(os.path.join(save_path, outname), "wb"),
    )



style_features = []
for i in range(len(style)):
    path = '../data/style_clip/' + style[i] + '.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print(data[style[i]].shape)
        style_features.append(data[style[i]])
