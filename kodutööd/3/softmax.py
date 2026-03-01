import torch

def softmax_loss_naive(W, X, y, reg):

    """

    Softmax (cross-entropy) loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A torch tensor of shape (D, C) containing weights.
    - X: A torch tensor of shape (N, D) containing a minibatch of data.
    - y: A torch tensor of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
        You might or might not want to transform it into one-hot form (not obligatory)
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    y = y.to(torch.long)

    W = W.to(torch.float64)
    X = X.to(torch.float64)

    loss = torch.tensor(0.0, dtype=torch.float64, device=W.device)
    dW = torch.zeros_like(W, dtype=torch.float64)
    num_train = X.shape[0]

    # In this naive implementation we have a for loop over the N samples
    for i in range(num_train):

        #############################################################################
        # TODO Task 3.1:
        # Compute the cross-entropy loss using explicit loops and store the   #
        # sum of losses in "loss". If you already understand the process well       #
        # and are familiar with vectorized operations, you can solve this task      #
        # without inner loops and use vectorized operations instead.                #
        # PS! But in this case still keep the outer loop that enumerates over X!    #
        # If you are not careful in implementing softmax, it is easy to run into    #
        # numeric instability, because exp(a) is huge if a is large.                #
        # Read the Practical issues: numeric stability section from here:           #
        # https://cs231n.github.io/linear-classify/#softmax-classifier              #
        #############################################################################

        # softmax function: f(x_i) = exp(x_i) / sum_j exp(x_j)
        z = X[i] @ W  # or .mm(W)
        z -= torch.max(z)
        exp_z = torch.exp(z)
        p = exp_z / torch.sum(exp_z)

        # loss: -log(p) of the correct class
        loss += -torch.log(p[y[i]])

        #############################################################################
        # TODO Task 3.3:                                                            #
        # Compute the gradient using explicit loops and store the sum over          #
        # samples in dW. Again, you are allowed to use vectorized operations        #
        # if you know how to.                                                       #
        #############################################################################

        # Compute gradient for sample i
        dW += X[i].unsqueeze(1) @ p.unsqueeze(0)  # shape (D, C)
        dW[:, y[i]] -= X[i]  # subtract X[i] for the correct class


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    # Average over training samples
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * torch.sum(W * W)
    dW += 2 * reg * W

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):


    """
    Softmax (cross-entropy) loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    W = W.to(torch.float64)
    X = X.to(torch.float64)
    y = y.to(torch.long)

    num_train = X.shape[0]

    #############################################################################
    # TODO Task 3.4:                                                            #
    # Compute the cross-entropy loss and its gradient using no loops.           #
    # Store the loss in loss and the gradient in dW.                            #
    # Make sure you take the average.                                           #
    # If you are not careful with softmax, you migh run into numeric instability#
    #############################################################################

    scores = X.mm(W)  # (N, C)

    # Numeric stability fix
    scores -= torch.max(scores, dim=1, keepdim=True)[0]

    # Softmax probabilities
    exp_scores = torch.exp(scores)
    probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)  # (N, C)

    # Cross-entropy loss
    correct_log_probs = -torch.log(probs[torch.arange(num_train), y])
    loss = torch.sum(correct_log_probs) / num_train

    # Gradient computation
    dscores = probs.clone()
    dscores[torch.arange(num_train), y] -= 1
    dscores /= num_train

    dW = X.t().mm(dscores)

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    # Add regularization to the loss and gradients.
    loss += reg * torch.sum(W * W)
    dW += 2 * reg * W

    return loss, dW