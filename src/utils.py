from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BillingualDataset, causal_mask
def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def sort_dataset_by_sentence_length(ds_raw, tokenizer_src, config):
    import numpy as np
    len_src_id = []
    sorted_list = []
    for item in ds_raw:
      src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
      len_src_id.append(len(src_ids))
    sorted_ids = np.argsort(len_src_id)
    sorted_list = [ds_raw[int(i)] for i in sorted_ids]
    return sorted_list

def get_dataset(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    src_lang = config["lang_src"]
    tgt_lang = config["lang_tgt"]
    seq_len = config["seq_len"]
    # sort sentences in dataset by length
    print("ds_raw", ds_raw.shape)

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, src_lang)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, tgt_lang)

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][src_lang]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][tgt_lang]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of the source sentence : {max_len_src}")
    print(f"Max length of the source target : {max_len_tgt}")

    ds_sorted = sort_dataset_by_sentence_length(ds_raw, tokenizer_src, config)
    print(f'Length of sorted dataset: {len(ds_sorted)}')
    print(f'Length of raw dataset: {len(ds_raw)}')
    train_ds_size = int(0.9 * len(ds_sorted))
    val_ds_size = len(ds_sorted) - train_ds_size
    train_ds_raw, val_ds_raw = ds_sorted[:train_ds_size], ds_sorted[train_ds_size:]

    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt


def get_dataloaders(config):
    from dataset import  CustomCollator
    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_dataset(config)
    custom_batch = CustomCollator(tokenizer_tgt=tokenizer_tgt)
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=custom_batch)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) # (1, 1) -> (bs, decoder_input)

    while True: # decoder predicts till we reach eos
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat( # concat next token to the current word
            [
                decoder_input,
                torch.empty(1, 1).type_as(source_mask).fill_(next_word.item()).to(device)
            ],
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0) # remove batch dim


def load_next_batch(val_dataloader, device, vocab_src, vocab_tgt, model, config):
    # Load a sample batch from the validation set
    batch = next(iter(val_dataloader))
    encoder_input = batch["encoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)

    encoder_input_tokens = [vocab_src.id_to_token(idx) for idx in encoder_input[0].cpu().numpy()]
    decoder_input_tokens = [vocab_tgt.id_to_token(idx) for idx in decoder_input[0].cpu().numpy()]

    # check that the batch size is 1
    assert encoder_input.size(
        0) == 1, "Batch size must be 1 for validation"

    model_out = greedy_decode(
        model, encoder_input, encoder_mask, vocab_src, vocab_tgt, config['seq_len'], device)

    return batch, encoder_input_tokens, decoder_input_tokens


