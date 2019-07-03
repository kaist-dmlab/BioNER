import torch

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
#     idxs.insert(0, 0)
#     idxs.append(1)
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_pp(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    idxs.insert(0, 0)
    idxs.append(1)
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def test_argmax():
    given = torch.rand(1, 5)
    assert argmax(given) == torch.argmax(given, 1).item()

def test():
    print("Test Start")
    test_argmax();
    test_prepare_sentence();
    print("Passed")

if __name__=="__main__":
    test()