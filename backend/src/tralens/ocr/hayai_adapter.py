"""Optional pinned local Hayai reader. Importing this module loads no ML runtime."""
import hashlib
import json
from pathlib import Path


REPOSITORY='JustANormalTinkerer/hayai-ocr-v2.7-supernova'
REVISION='517bdeeb64b68c08939e12f7211e99943c1e1640'
FILES={
    'config.json':'e5cf0e59b888ef29155edb40b28f42184b1e22523fca3b38ba62261e743cb91b',
    'configuration_hayai.py':'bbff66c6a1bb7ee2f8097359992546e1c440b65285c608c22c41b204d13927fe',
    'modeling_hayai.py':'cb78b50775a049e066ceb501667228c471f10365d66fb3f0de019b76ca3017a6',
    'processing_hayai.py':'047db0d41055d44322a0e446d3fa8aa3cfb5525ef0d59351e8e8fa2e1443b484',
    'tokenizer.json':'f8a0a909c628a684fe463094614e236a8b1d3609e7770f77e7beafaf1056bf13',
    'tokenizer_config.json':'6fb6c69afaedf1275872d3e62e276fd4467bd00da7a84cbbb5566a2cd28f58f6',
    'model.safetensors':'c7dd04f7138e74abb48ea0dcf6bad7f2bdedd704fe5f8ec6328962e799c65837',
    'vision-config.json':'c0b8c2e7f0527b0bea1b1d9abe0381c0f294352df92439c54590b8420e539118',
}


def verify_snapshot(model_path):
    path=Path(model_path).expanduser().resolve(strict=True)
    if not path.is_dir():raise ValueError('Hayai requires an existing pinned local snapshot')
    for name,expected in FILES.items():
        asset=path/name
        limit=640*1024*1024 if name=='model.safetensors' else 2*1024*1024
        if not asset.is_file() or asset.stat().st_size>limit:raise ValueError('missing/oversized Hayai asset: '+name)
        checksum=hashlib.sha256()
        with asset.open('rb') as stream:
            while chunk:=stream.read(1024*1024):checksum.update(chunk)
        if checksum.hexdigest()!=expected:raise ValueError('pinned Hayai asset mismatch: '+name)
    return path


class HayaiReader:
    def __init__(self,model_path):
        path=verify_snapshot(model_path)
        import os
        # The reviewed custom modules contain compilation decorators. This local
        # CPU route uses eager kernels and never requests a Hub artifact.
        settings=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HAYAI_DISABLE_COMPILE='1')
        self.previous_environment={key:os.environ.get(key) for key in settings}
        self.previous_threads=None;self.model=None;self.processor=None
        os.environ.update(settings)
        try:
            import torch
            from importlib.metadata import version
            from safetensors.torch import load_file
            from transformers import AutoConfig,AutoModel,PreTrainedTokenizerFast
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            self.previous_threads=torch.get_num_threads()
            torch.set_num_threads(4)
            config=AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
            config.vision_config=json.loads((path/'vision-config.json').read_bytes())['vision_config']
            model=AutoModel.from_config(config,trust_remote_code=True)
            weights=load_file(path/'model.safetensors',device='cpu')
            loaded=model.load_state_dict(weights,strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:raise ValueError('Hayai checkpoint architecture mismatch')
            del weights
            model.eval()
            tokens=json.loads((path/'tokenizer_config.json').read_bytes())
            tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(path/'tokenizer.json'),
                **{k:tokens[k] for k in ('bos_token','eos_token','pad_token','unk_token')})
            processor=get_class_from_dynamic_module('processing_hayai.HayaiProcessor',str(path),local_files_only=True)
            self.processor=processor(tokenizer=tokenizer);self.model=model
            self.identity={'repository':REPOSITORY,'revision':REVISION,'sha256':FILES.copy(),
                'vision_config_repository':'google/siglip2-base-patch16-naflex',
                'vision_config_revision':'b53b807d3a2d5e2b3911292f2d69e5341cdc064c',
                'runtime':{name:version(name) for name in ('torch','transformers','tokenizers','safetensors')},
                'device':'cpu','dtype':'float32','threads':4,'max_patches':256,'max_new_tokens':128,
                'tokenizer_loader':'unchanged JSON via PreTrainedTokenizerFast','downloads':False}
        except BaseException:
            self.close();raise

    def __call__(self,image):
        if self.model is None:raise ValueError('Hayai reader is closed')
        if image.mode!='RGB' or max(image.size)>1040 or image.width*image.height>300000:
            raise ValueError('Hayai requires a bounded RGB crop')
        import torch
        return self.model.ocr(image,self.processor,max_patches=256,max_new_tokens=128,
            use_cuda_graph=False,dtype=torch.float32)

    def close(self):
        import os
        self.model=None;self.processor=None
        if self.previous_threads is not None:
            import torch
            torch.set_num_threads(self.previous_threads);self.previous_threads=None
        for key,value in self.previous_environment.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        self.previous_environment={}
