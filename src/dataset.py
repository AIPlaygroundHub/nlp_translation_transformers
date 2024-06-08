import torch
from torch.utils.data import Dataset


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class BillingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds # list of dicts [{'id': '8300', 'translation' : {'en' : ..., 'it' : ...}}, ...]
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        return {
            "enc_input_tokens": enc_input_tokens,
            "dec_input_tokens": dec_input_tokens,
            # "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # # encoder mask: (1, 1, seq_len) -> Has 1 when there is text and 0 when there is pad (no text)

            # "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            # # (1, seq_len) and (1, seq_len, seq_len)
            # # Will get 0 for all pads. And 0 for earlier text.
            "target_tokens": dec_input_tokens,
            "src_text": src_text,
            "tgt_text": tgt_text,

            }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal = 1).type(torch.int)
    #This will get the upper traingle values
    return mask == 0


class CustomCollator:
    def __init__(self, **params):
        tokenizer_tgt = params['tokenizer_tgt']
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __call__(self, batch):
        print("CustomCollator is called")
        max_len_enc = max(len(x["enc_input_tokens"]) for x in batch)
        max_len_dec = max(len(x["dec_input_tokens"]) for x in batch)

        encoder_inputs = []
        decoder_inputs = []
        encoder_masks = []
        decoder_masks = []
        targets = []
        src_texts = []
        tgt_texts = []

        for b in batch:
            enc_input_tokens = b['enc_input_tokens']
            dec_input_tokens = b['dec_input_tokens']
            target_tokens = b['target_tokens']
            enc_num_padding_tokens = max_len_enc - len(b["enc_input_tokens"])
            dec_num_padding_tokens = max_len_dec - len(b["dec_input_tokens"])
            # print(f'Adding {enc_num_padding_tokens} pad tokens to sentence of length {len(b["enc_input_tokens"])}.')

            encoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
                ],
                dim=0,
            )

            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
                ],
                dim=0,
            )

            target = torch.cat(
                [
                    torch.tensor(dec_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )

            encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            # print("encoder_input shape in ds", encoder_input.shape)
            # print("mask shape", (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).shape)
            encoder_masks.append((encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int())
            decoder_masks.append((decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)))

            targets.append(target)

            src_texts.append(b["src_text"])
            tgt_texts.append(b["tgt_text"])

        return {
            "encoder_input": torch.vstack(encoder_inputs),
            "decoder_input": torch.vstack(decoder_inputs),
            "encoder_mask": torch.vstack(encoder_masks),
            "decoder_mask": torch.vstack(decoder_masks),
            "target": torch.vstack(targets),
            "src_text": src_texts,
            "tgt_text": tgt_texts,
        }
    
    
    