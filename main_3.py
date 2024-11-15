# Код частично взят из https://pytorch.org/tutorials/beginner/translation_transformer.html

from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from tqdm import tqdm
import torch
import os
# import evaluate
# import wandb
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)

# enable_logging = True

# metric = evaluate.load('sacrebleu')

def build_and_run_model(model_type):
    print(model_type)

    # if enable_logging:
    #     wandb.login()
    #     wandb.init(project="bhw2-Ivanova")

    if model_type == 'en-de':
        files = {'train_de' : 'train.de-en.de',
                'train_en' : 'train.de-en.en',
                'val_de' : 'val.de-en.de',
                'val_en' : 'val.de-en.en'}
    else:
        files = {'train_de' : 'train1.de-en.de',
                'train_en' : 'train1.de-en.en',
                'val_de' : 'val.de-en.de',
                'val_en' : 'val.de-en.en',
                'test_de' : 'test1.de-en.de'}

    class TextDataset(Dataset):
        def __init__(self, root, train=True, val=False, dataset_size=None):
            super().__init__()
            # dataset_size=10
            self.root = root
            self.train = train
            self.val = val
            self.de_strings = []
            self.en_strings = []
            
            filename = ''
            if train:
                filename = 'train'
            elif val:
                filename = 'val'
            else:
                filename = 'test'

            f_de = open(os.path.join(self.root, files[filename + '_de']))
            for line in f_de:
                self.de_strings.append(line[:-1])
            f_de.close()
            if dataset_size is not None and len(self.de_strings) > dataset_size:
                self.de_strings = self.de_strings[:dataset_size]
            if self.train or self.val:
                f_en = open(os.path.join(self.root, files[filename + '_en']))
                for line in f_en:
                    self.en_strings.append(line[:-1])
                f_en.close()
            if dataset_size is not None and len(self.en_strings) > dataset_size:
                self.en_strings = self.en_strings[:dataset_size]

        def __len__(self):
            return len(self.de_strings)

        def __getitem__(self, item):
            de_string = ''
            en_string = ''
            if len(self.de_strings) != 0:
                de_string = self.de_strings[item]
            if len(self.en_strings) != 0:
                en_string = self.en_strings[item]
            if model_type == 'en-de':
                return en_string, de_string
            return de_string, en_string

    if model_type == 'en-de':
        SRC_LANGUAGE = 'en'
        TGT_LANGUAGE = 'de'
    else:
        SRC_LANGUAGE = 'de'
        TGT_LANGUAGE = 'en'

    # Place-holders
    token_transform = {}
    vocab_transform = {}


    # Create source and target language tokenizer.
    token_transform[SRC_LANGUAGE] = get_tokenizer(None, language=SRC_LANGUAGE)
    token_transform[TGT_LANGUAGE] = get_tokenizer(None, language=TGT_LANGUAGE)

    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index[language]])

    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = TextDataset(root='', train=True, val=False, dataset_size=None)
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=10,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)
        print('vocab len', ln, len(vocab_transform[ln]))

    # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
    class PositionalEncoding(nn.Module):
        def __init__(self,
                    emb_size: int,
                    dropout: float,
                    maxlen: int = 5000):
            super(PositionalEncoding, self).__init__()
            den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
            pos = torch.arange(0, maxlen).reshape(maxlen, 1)
            pos_embedding = torch.zeros((maxlen, emb_size))
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)
            pos_embedding = pos_embedding.unsqueeze(-2)

            self.dropout = nn.Dropout(dropout)
            self.register_buffer('pos_embedding', pos_embedding)

        def forward(self, token_embedding: Tensor):
            return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

    # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size: int, emb_size):
            super(TokenEmbedding, self).__init__()
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.emb_size = emb_size

        def forward(self, tokens: Tensor):
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

    # Seq2Seq Network
    class Seq2SeqTransformer(nn.Module):
        def __init__(self,
                    num_encoder_layers: int,
                    num_decoder_layers: int,
                    emb_size: int,
                    nhead: int,
                    src_vocab_size: int,
                    tgt_vocab_size: int,
                    dim_feedforward: int = 512,
                    dropout: float = 0.1):
            super(Seq2SeqTransformer, self).__init__()
            self.transformer = Transformer(d_model=emb_size,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout)
            self.generator = nn.Linear(emb_size, tgt_vocab_size)
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
            self.positional_encoding = PositionalEncoding(
                emb_size, dropout=dropout)

        def forward(self,
                    src: Tensor,
                    trg: Tensor,
                    src_mask: Tensor,
                    tgt_mask: Tensor,
                    src_padding_mask: Tensor,
                    tgt_padding_mask: Tensor,
                    memory_key_padding_mask: Tensor):
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
            outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                    src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
            return self.generator(outs)

        def encode(self, src: Tensor, src_mask: Tensor):
            return self.transformer.encoder(self.positional_encoding(
                                self.src_tok_emb(src)), src_mask)

        def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
            return self.transformer.decoder(self.positional_encoding(
                            self.tgt_tok_emb(tgt)), memory,
                            tgt_mask)

    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                    NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


    # helper function to club together sequential operations
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                        torch.tensor(token_ids),
                        torch.tensor([EOS_IDX])))

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor


    # function to collate data samples into batch tesors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch


    def train_epoch(model, optimizer):
        model.train()
        losses = 0
        train_iter = TextDataset(root='', train=True, val=False, dataset_size=None)
        train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        for src, tgt in tqdm(train_dataloader, 'train'):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)


    def evaluate(model):
        model.eval()
        losses = 0

        val_iter = TextDataset(root='', train=False, val=True, dataset_size=None)
        val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        for src, tgt in tqdm(val_dataloader, 'val'):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)

    # function to generate output sequence using greedy algorithm
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        src = src.to(DEVICE)
        src_mask = src_mask.to(DEVICE)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys


    # actual function to translate input sentence into target language
    def translate(model: torch.nn.Module, src_sentence: str):
        model.eval()
        src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    if model_type == 'en-de':
        NUM_EPOCHS = 20
    else:
        NUM_EPOCHS = 20
    calc_bleu = 5

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        # if enable_logging:
        #     wandb.log({"train loss" : train_loss, "val loss" : val_loss})

        if epoch % calc_bleu == 0:
            train_iter = TextDataset(root='', train=True, val=False, dataset_size=100)
            val_iter = TextDataset(root='', train=False, val=True, dataset_size=100)
            train_pred = []
            train_target = []
            for de_string, en_string in train_iter:
                train_pred.append(translate(transformer, de_string))
                train_target.append(en_string)
            val_pred = []
            val_target = []
            for de_string, en_string in val_iter:
                val_pred.append(translate(transformer, de_string))
                val_target.append(en_string)
            # train_score = metric.compute(predictions=train_pred, references=train_target)['score']
            # val_score = metric.compute(predictions=val_pred, references=val_target)['score']
            # print((f"Train BLEU: {train_score:.3f}, Val BLEU: {val_score:.3f}"))
            # if enable_logging:
            #     wandb.log({"train BLEU" : train_score, "val BLEU" : val_score})

            if model_type == 'de-en':
                f = open('ans', 'w')
                test_iter = TextDataset(root='', train=False, val=False, dataset_size=None)
                for de_string, _ in tqdm(test_iter, 'translate de-en'):
                    f.write(translate(transformer, de_string))
                    f.write('\n')
                f.close()

        # if enable_logging:
        #     wandb.watch(transformer)

    if model_type == 'en-de':
        f_en = open('train1.de-en.en', 'w')
        f_de = open('train1.de-en.de', 'w')
        train_iter = TextDataset(root='', train=True, val=False, dataset_size=None)
        for en_string, de_string in tqdm(train_iter, 'translate en-de'):
            f_en.write(en_string)
            f_en.write('\n')
            f_en.write(en_string)
            f_en.write('\n')
            f_de.write(de_string)
            f_de.write('\n')
            f_de.write(translate(transformer, en_string))
            f_de.write('\n')
        f_en.close()
        f_de.close()
    # test_iter = TextDataset(root='', train=False, val=False, dataset_size=None)
    # f = open('drive/MyDrive/bhw2/bhw2_answer_exp2', 'w')
    # for de_string, _ in tqdm(test_iter, 'eval'):
    #   f.write(translate(transformer, de_string))
    #   f.write('\n')
    # f.close()

build_and_run_model('en-de')
build_and_run_model('de-en')