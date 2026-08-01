import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip.clip import _download, _MODELS, build_model_from_openai_state_dict, tokenize

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.factory import create_model


_tokenizer = _Tokenizer()

# _MODELS = {
#     "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
#     "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
#     "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
#     "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
#     "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
#     "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
#     "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
#     "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
#     "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
# }

def load_clip_to_cpu(args):
    backbone_name = args.clip_backbone
    url = _MODELS[backbone_name]
    model_path = _download(url, root="./models/clip_download")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()

        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 2,
                      "language_depth": 2, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": args.maple_train_n_ctx}
    model = build_model_from_openai_state_dict(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.visual.conv1.weight.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        '''
        prompts:
        tokenized_prompts:
        compound_prompts_deeper_text:
        '''
        x = prompts + self.positional_embedding.type(self.dtype)  # add position encode
        x = x.permute(1, 0, 2)  # NLD -> LND #n_tkn, n_cls, c
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pas
        #combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        #print(x.shape, compound_prompts_deeper_text[0].shape, len(compound_prompts_deeper_text)) #torch.Size([77, 89, 512]) torch.Size([2, 512]) 2
        for deeper_text in compound_prompts_deeper_text:
            deeper_text = deeper_text.unsqueeze(1).repeat(1, x.shape[1], 1) #2 89 512
            new_ctx = x[1:3] + deeper_text
            combined = torch.cat((x[:1], new_ctx, x[3:]), dim=0)
        outputs = self.transformer(combined) #[77, 89, 512]
        # x = outputs[0]  # extract the x back from here
        x = outputs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # normalize

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        classnames = args.classnames
        n_cls = len(classnames) #dim0 is either batch_size (during training) or n_cls (during testing)
        n_ctx = args.maple_train_n_ctx  # 上下文长度
        ctx_init = args.maple_train_ctx_init  # 用于初始化上下文向量的初始文本
        dtype = clip_model.visual.conv1.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # print(clip_imsize, 'clip_imgsize')
        # cfg_imsize = [args.max_size, 320]
        # Default is 1, which is compound shallow prompting
        assert args.maple_train_prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.maple_train_prompt_depth  # max=12, but will create 11 such shared prompts
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = tokenize(ctx_init)  # 使用 CLIP 的 tokenize 方法将 ctx_init 转换为 token
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # 从 embedding 中提取上下文向量 ctx_vectors
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        #self.proj.half()  # 线性层的所有权重和偏置参数转换为半精度浮点数
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)  # 使用正态分布初始化这些参数，标准差为 0.02
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) #[89, 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim) [89, 1, 512]
                ctx,  # (dim0, n_ctx, dim) [89, 2, 512]
                suffix,  # (dim0, *, dim) [89, 74, 512]
            ],
            dim=1,
        )

        return prompts  # [89, 77, 512]

    def forward(self):
        ctx = self.ctx  # nn.Parameter(ctx_vectors)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class MaPLe(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()

        self.prompt_learner = MultiModalPromptLearner(args, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.label_linear = nn.Linear(256, len(args.classnames))

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        # print(image.shape) #[5, 3, 160, 224]
        # print(shared_ctx.shape) #[2, 768]
        #print(text_features.shape) #89 512
        # print(deep_compound_prompts_text[0].shape, len(deep_compound_prompts_text[0])) #[2, 512] 2 image.type(self.dtype)
        image_resize = F.interpolate(
                image.type(self.dtype),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
        image_features = self.image_encoder(image_resize)
        # print(image_features.shape) #[5, 512]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        #print(logits.shape) #5 89
        label = label / label.norm(dim=1, keepdim=True)
        label = self.label_linear(label)

        if self.prompt_learner.training:
            return torch.matmul(logits, label.T).mean(), image_features, text_features

        return logits, image_features, text_features

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_maple(args):
    assert args.maple_prec in ["fp16", "fp32", "amp"]

    #init clip model
    print(f"Loading CLIP (backbone: {args.clip_backbone})")
    clip_model = load_clip_to_cpu(args)
    if args.maple_prec == "fp32" or args.maple_prec == "amp":
        # CLIP's default precision is fp16
        clip_model.float()

    #init maple
    print("Building MaPLe")
    maple_model = MaPLe(args, clip_model)

    return maple_model