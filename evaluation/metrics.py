# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertModel
from typing import List
import numpy as np
from scipy.stats import entropy


class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))


import numpy as np
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import pandas as pd


class DomainSpecificity:
    def __init__(self):
        self.model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    def cosine_similarity(self, vecs):
        vec1 = vecs[0]
        vec2 = vecs[1]
        return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def embed(self, sentences):
        return self.model.encode(sentences)

    def caluate(self, q1, q2):
        return self.cosine_similarity(self.embed([q1, q2]))


class SemanticEntropy:

    def __init__(self):
        self.tokenizer, self.model = self.load_bert_model()

    def load_bert_model(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
        return tokenizer, model

    def get_bert_embedding(self, tokenizer, model, text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states[-1]
        sentence_embedding = torch.mean(hidden_states, dim=1)
        return sentence_embedding.squeeze()

    def cosine_similarity(self, vec1, vec2):
        cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
        return cos_sim.item()

    def calculate_entropy(self, similarity):
        # Calculate entropy based on similarity and dissimilarity
        p_sim = similarity
        p_diff = 1 - similarity
        probabilities = torch.tensor([p_sim, p_diff])
        return entropy(probabilities.numpy(), base=2)

    def calculate_semantic_entropy(self, vignette, context):
        embedding1 = self.get_bert_embedding(self.tokenizer, self.model, vignette)
        embedding2 = self.get_bert_embedding(self.tokenizer, self.model, context)
        similarity_score = self.cosine_similarity(embedding1, embedding2)
        entropy_score = self.calculate_entropy(similarity_score)
        return entropy_score, similarity_score
