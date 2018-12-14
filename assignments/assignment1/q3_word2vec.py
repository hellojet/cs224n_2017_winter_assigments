#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    # get the sum of the rows of x**2, the x / what we get just now to normalization
    x = x / np.sqrt(np.sum(x**2, 1)).reshape(-1,1)
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    '''print variables:
    print predicted.shape: 3x
    print target: 4
    print outputVectors.shape: 5x3
    end print'''
    # get y: y[target] = 1, else position = 0
    # y: 1x5
    y = np.zeros((1, outputVectors.shape[0]))
    y[0,target] = 1
    # get y_hat
    # outputVectors is U matrix, predicted is vector v
    # y_hat: 5x
    y_hat = softmax(np.dot(outputVectors, predicted)).T
    # get cost: CE = -y*log(y_hat)
    cost = -y.dot(np.log(y_hat))
    # get gradPred: U(y_hat-y)
    # gradPred: 3x
    gradPred = (outputVectors.T.dot(np.transpose(y_hat - y))).flatten()
    # get grad: v(y_hat-y).T
    # grad: 5x3
    grad = (y_hat - y).T.dot(predicted.reshape(1,-1))
    ### END YOUR CODE
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    grad = np.zeros(outputVectors.shape)
    # get cost: -log(sigmoid(Uv))-\Sigma(log(sigmoid(-u[indices]v)))
    # predicted: 3x outputVectors:5x3
    z = sigmoid(outputVectors[target].dot(predicted))
    cost = -np.log(z)
    # get gradPred: (sigmoid(uv)-1)u-\Sigma((sigmoid(-u_kv)-1)u)
    gradPred = (z-1.0) * outputVectors[target]
    # get grad[target]: (sigmoid(uv)-1)v
    # get grad[k]: -(sigmoid(-u_k * v)-1)v
    grad[target] = (z - 1.0) * predicted

    for k in xrange(K):
        samp = dataset.sampleTokenIdx()
        z = sigmoid(np.dot(outputVectors[samp], predicted))
        cost -= np.log(1.0 - z)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z
    # for indice in indices:
    #     cost -= np.log(sigmoid(-outputVectors[indice,:].dot(predicted)))
    #     gradPred -= (sigmoid(-outputVectors[indice].dot(predicted))-1) * outputVectors[indice]
    #     grad[indice] += -(sigmoid(-outputVectors[indice].reshape(1,-1).dot(predicted)) - 1)*predicted
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    ''' print information:
    currentWord: b
    C: 3
    contextWords: ['e', 'a', 'c', 'c', 'd', 'd']
    tokens: {'a': 0, 'c': 2, 'b': 1, 'e': 4, 'd': 3}
    inputVectors: 5x3 [[-0.96735714 -0.02182641  0.25247529]
    [ 0.73663029 -0.48088687 -0.47552459]
    [-0.27323645  0.12538062  0.95374082]
    [-0.56713774 -0.27178229 -0.77748902]
    [-0.59609459  0.7795666   0.19221644]]
    outputVectors: 5x3 [[-0.6831809  -0.04200519  0.72904007]
    [ 0.18289107  0.76098587 -0.62245591]
    [-0.61517874  0.5147624  -0.59713884]
    [-0.33867074 -0.80966534 -0.47931635]
    [-0.52629529 -0.78190408  0.33412466]]
    end print'''
    c_word_index = tokens[currentWord]
    v_hat = inputVectors[c_word_index]
    for j in contextWords:
        o_context_index = tokens[j]
        t_cost, t_gradIn, t_gradOut = word2vecCostAndGradient(v_hat, o_context_index, outputVectors, dataset)
        cost += t_cost
        gradIn[c_word_index] += t_gradIn
        gradOut += t_gradOut
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    ### YOUR CODE HERE
    v_hat = np.zeros((inputVectors.shape[1],))
    for j in contextWords:
        context_index = tokens[j]
        v_hat += inputVectors[context_index]
    cost, c_gradIn, gradOut = word2vecCostAndGradient(v_hat, tokens[currentWord], outputVectors, dataset)
    for j in contextWords:
        gradIn[tokens[j]] += c_gradIn # here '+=' means that repeated words may appear in the context
    # gradIn[tokens[currentWord]]
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    ''' print variables:
    print grad: 10x3 zeros
    print N: 10
    print inputVectors: 5x3
    [[-0.96735714 -0.02182641  0.25247529]
    [ 0.73663029 -0.48088687 -0.47552459]
    [-0.27323645  0.12538062  0.95374082]
    [-0.56713774 -0.27178229 -0.77748902]
    [-0.59609459  0.7795666   0.19221644]]
    print outputVectors: 5x3
    [[-0.6831809  -0.04200519  0.72904007]
    [ 0.18289107  0.76098587 -0.62245591]
    [-0.61517874  0.5147624  -0.59713884]
    [-0.33867074 -0.80966534 -0.47931635]
    [-0.52629529 -0.78190408  0.33412466]]
    print C: 5
    '''
    # traverse from 0 to batchsize - 1(that is 49)
    for i in xrange(batchsize):
        # select the centor word C1 from [1, 5]
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    # return integer between (0,4], used for negative sampling
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    # set the random seed for fixing the random number/matrix after, and convenience for testing
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="

    # test skip-gram and softmax
    ''' print variables:
    dummy_tokens: {'a': 0, 'c': 2, 'b': 1, 'e': 4, 'd': 3}
    dummy_vectors: 10x3
    [[-0.96735714 -0.02182641  0.25247529]
    [ 0.73663029 -0.48088687 -0.47552459]
    [-0.27323645  0.12538062  0.95374082]
    [-0.56713774 -0.27178229 -0.77748902]
    [-0.59609459  0.7795666   0.19221644]
    [-0.6831809  -0.04200519  0.72904007]
    [ 0.18289107  0.76098587 -0.62245591]
    [-0.61517874  0.5147624  -0.59713884]
    [-0.33867074 -0.80966534 -0.47931635]
    [-0.52629529 -0.78190408  0.33412466]]
    '''
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    # test CBOW and softmax
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    # test CBOW and negative sampling
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
