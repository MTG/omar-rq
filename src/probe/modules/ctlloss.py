import numpy as np
import torch
from torch.autograd import Variable
from functools import reduce


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def tensor(array):
    if array.dtype == 'bool':
        array = array.astype('uint8')
    return cuda(torch.from_numpy(array))


def variable(array):
    if isinstance(array, np.ndarray):
        array = tensor(array)
    return cuda(array)


def logsumexp(*args):
    M = reduce(torch.max, args)
    mask = M != -np.inf
    M[mask] += torch.log(sum(torch.exp(x[mask] - M[mask]) for x in args))
    return M


def ctl_loss(frameProb, label, maxConcur=1, debug=False):
    """
    Compute the Connectionist Temporal Loss (CTL) for sequence labeling problems without seqLen (no padding handling).

    Arguments:
    - frameProb: A 3D tensor of shape [N_SEQS x N_FRAMES x N_CLASSES] containing the probability of each event
                 occurring at each frame for each sequence.
    - label: A list of label sequences, each containing the sequence of class labels (events) expected for each frame.
    - maxConcur: (Optional) Maximum number of events that can be emitted simultaneously at any frame.
                 Default is 1 (no concurrency).
    - debug: (Optional) If set to True, the function also returns per-sequence log probabilities along with the total loss.

    The function performs the following steps:
    1. Converts frame-wise event probabilities into boundary probabilities (start and end of events).
    2. Computes log probabilities for not emitting any event (blankLogProb) and for emitting each event (deltaLogProb).
    3. Initializes an alpha trellis to compute the accumulated log probability of emitting tokens (events)
       across the sequence using dynamic programming.
    4. Iteratively updates the alpha matrix for each frame, considering the cases where no event is emitted and
       where multiple events (up to maxConcur) are emitted concurrently.
    5. Collects the final log probability for each sequence.
    6. Computes the total loss as the negative log probability of all utterances (sequences).

    Returns:
    - If debug is False: the average loss across sequences.
    - If debug is True: the average loss and the log probability of each individual sequence.
    """

    nSeqs, nFrames, nClasses = frameProb.size()

    # Convert frameProb (probabilities of events) into probabilities of event boundaries
    z = variable(1e-7 * torch.ones((nSeqs, 1, nClasses)))  # Small non-zero value to avoid NaNs
    frameProb = torch.cat([z, frameProb, z], dim=1)
    startProb = torch.clamp(frameProb[:, 1:] - frameProb[:, :-1], min=1e-7)
    endProb = torch.clamp(frameProb[:, :-1] - frameProb[:, 1:], min=1e-7)
    boundaryProb = torch.stack([startProb, endProb], dim=3).view((nSeqs, nFrames + 1, nClasses * 2))

    blankLogProb = torch.log(1 - boundaryProb).sum(dim=2)
    deltaLogProb = torch.log(boundaryProb) - torch.log(1 - boundaryProb)

    # Find out the lengths of the label sequences
    labelLen = tensor(np.array([len(x) for x in label]))

    # Put the label sequences into a Variable
    maxLabelLen = max(len(x) for x in label)
    L = np.zeros((nSeqs, maxLabelLen), dtype='int64')
    for i in range(nSeqs):
        L[i, :len(label[i])] = np.array(label[i]) - 1  # Adjust labels to match index range
    label = tensor(L)

    if maxConcur > maxLabelLen:
        maxConcur = maxLabelLen

    # Compute alpha trellis
    nStates = maxLabelLen + 1
    alpha = variable(-np.inf * torch.ones((nSeqs, nStates)))
    alpha[:, 0] = 0
    seqIndex = tensor(np.tile(np.arange(nSeqs), (nStates, 1)).T)
    dummyColumns = variable(-np.inf * torch.ones((nSeqs, maxConcur)))
    uttLogProb = variable(torch.zeros(nSeqs))

    for frame in range(nFrames + 1):  # +1 because we are considering boundaries
        # Case 0: don't emit anything at current frame
        p = alpha + blankLogProb[:, frame].view((-1, 1))
        alpha = p
        for i in range(1, maxConcur + 1):
            # Case i: emit i tokens at current frame
            p = p[:, :-1] + deltaLogProb[seqIndex[:, i:], frame, label[:, (i - 1):]]
            alpha = logsumexp(alpha, torch.cat([dummyColumns[:, :i], p], dim=1))

        # Collect probability for ends of utterances
        uttLogProb = alpha[:, labelLen].clone()

    # Return the negative log probability of all utterances (and per-utterance log probs if debug == True)
    loss = -uttLogProb.sum() / nSeqs
    if debug:
        return loss, uttLogProb
    else:
        return loss


if __name__ == '__main__':
    def strip(variable):
        return variable.data.cpu().numpy()


    torch.set_printoptions(precision=5)

    frameProb = np.array([[[0.1, 0.9, 0.9], [0.1, 0.9, 0.9], [0.1, 0.9, 0.9], [0.1, 0.9, 0.1]]], dtype='float32')
    frameProb = np.tile(frameProb, (4, 1, 1))
    frameProb = Variable(tensor(frameProb), requires_grad=True)
    label = [[3, 5, 6, 4], [3, 4], [5, 6], []]

    loss, uttLogProb = ctl_loss(frameProb, label, maxConcur=1, debug=True)
    print(strip(loss), strip(torch.exp(uttLogProb)))

    loss, uttLogProb = ctl_loss(frameProb, label, maxConcur=2, debug=True)
    print(strip(loss), strip(torch.exp(uttLogProb)))

    loss, uttLogProb = ctl_loss(frameProb, label, maxConcur=3, debug=True)
    print(strip(loss), strip(torch.exp(uttLogProb)))

    loss.backward()
    print(frameProb.grad)
