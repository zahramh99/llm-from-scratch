import torch

def tokenize(text, vocab):
    """Tokenize text into vocabulary IDs.
    
    Args:
        text: Input string to tokenize
        vocab: Dictionary mapping words to IDs
        
    Returns:
        List of token IDs
    """
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]

def build_vocab(texts, special_tokens=["<UNK>"]):
    """Build vocabulary from list of texts.
    
    Args:
        texts: List of text strings
        special_tokens: List of special tokens to add
        
    Returns:
        vocab: Dictionary mapping words to IDs
    """
    vocab = {}
    # Add special tokens first
    for token in special_tokens:
        vocab[token] = len(vocab)
    
    # Add words from texts
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    return vocab