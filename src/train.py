import warnings
import torch
import torch.nn as nn
import torchmetrics
from config_file import get_config, get_weights_file_path
from utils import get_dataloaders, greedy_decode
from model import build_transformer

from pathlib import Path
from tqdm import tqdm
import wandb
# wandb.init(project="nlp_translation_transformers", config=get_config())

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'],config['seq_len'], config['d_model'] )
    return model

def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, global_step, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # convert tokens to text

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            if count <= num_examples:
                print("SOURCE", source_text)
                print("TARGET", target_text)
                print("PREDICTED", model_out_text)

            cer_metric = torchmetrics.CharErrorRate()
            cer = cer_metric(predicted, expected)
            wer_metric = torchmetrics.WordErrorRate()
            wer = wer_metric(predicted, expected)
            bleu_metric = torchmetrics.BLEUScore()
            bleu = bleu_metric(predicted, expected)
            # wandb.log({'character error rate': cer, 'Word Error Rate': wer, 'BLEUScore': bleu})

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (bs, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (bs, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (bs, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (bs, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (bs, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (bs, seq_len, d_model)
            proj_output = model.project(decoder_output) # (bs, seq_len, tgt_vocab_size)

            label = batch['target'].to(device) # (bs, seq_len)

            # view transforms (bs, seq_len, tgt_vocab_size) to (bs * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # wandb.log({"train_loss": loss.item()})

            loss.backward()
            optimizer.step() # update weights
            optimizer.zero_grad()

            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, global_step)
        # model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # torch.save({'model_state_dict': model.state_dict(),
        #             'epoch' : epoch,
        #             'optimizer_state_dict' : optimizer.state_dict(),
        #             'global_step' : global_step
        #             }, model_filename)






if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)