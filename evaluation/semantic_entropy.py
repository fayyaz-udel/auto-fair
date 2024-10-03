import torch
from transformers import BertTokenizer, BertModel
from scipy.stats import entropy


def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, return_dict=True)
    return tokenizer, model


def get_bert_embedding(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[-1]
    sentence_embedding = torch.mean(hidden_states, dim=1)
    return sentence_embedding.squeeze()


def cosine_similarity(vec1, vec2):
    cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
    return cos_sim.item()


def calculate_entropy(similarity):
    # Calculate entropy based on similarity and dissimilarity
    p_sim = similarity
    p_diff = 1 - similarity
    probabilities = torch.tensor([p_sim, p_diff])
    return entropy(probabilities.numpy(), base=2)


def calculate_semantic_entropy(vignette, context):
    tokenizer, model = load_bert_model()
    embedding1 = get_bert_embedding(tokenizer, model, vignette)
    embedding2 = get_bert_embedding(tokenizer, model, context)
    similarity_score = cosine_similarity(embedding1, embedding2)
    entropy_score = calculate_entropy(similarity_score)
    return entropy_score, similarity_score