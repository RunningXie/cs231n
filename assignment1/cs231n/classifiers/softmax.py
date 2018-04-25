import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    f = X[i].dot(W)#shape=(C,)
    f -= np.max(f)  # 剪掉每列的最大值防止值过大
    loss += np.log(np.sum(np.exp(f))) - f[y[i]]
    dW[:,y[i]]-=X[i]
    for j in xrange(num_classes):
        dW[:, j] += np.exp(f[j])*X[i]/np.sum(np.exp(f))#这里log没写底数默认为e，(e^x)'=e^x,(logx)'=1/(xlna)
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  f = X.dot(W)#shape=(N,C)
  f -= np.max(f,axis=1).reshape(num_train,1)  # 剪掉每列的最大值防止值过大
  s=np.sum(np.exp(f),axis=1)
  loss += np.sum(np.log(s)) - np.sum(f[range(num_train),y])
  loss /= num_train
  loss += reg * np.sum(W * W)
  counts = np.exp(f) /s.reshape(num_train,1)
  counts[range(num_train), y] -= 1
  dW = np.dot(X.T, counts)
  dW/=num_train
  dW += 2 * reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

