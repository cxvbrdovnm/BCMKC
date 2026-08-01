import torch
import clip

def text_prompt(data):
    # text_aug = ['{}']
    text_aug = 'This is a video about {}'
    classes = torch.cat([clip.tokenize(text_aug.format(c)) for c in data])
    # classes = clip.tokenize(text_aug.format(data))
    return classes