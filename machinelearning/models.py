from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
import torch
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, ones, matmul
from torch.nn.functional import cross_entropy, relu, mse_loss, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        return self.w

    def run(self, x):
        # score = dot product of weights and input
        return matmul(x, self.w.T)

    def get_prediction(self, x):
        return 1 if self.run(x).item() >= 0 else -1

    def train(self, dataset):
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            while True:
                mistakes = 0
                for batch in dataloader:
                    x = batch['x']
                    y = batch['label'].item()
                    if self.get_prediction(x) != y:
                        self.w += y * x
                        mistakes += 1
                if mistakes == 0:
                    break


class RegressionModel(Module):
    """
    A neural network model for approximating sin(x) on [-2pi, 2pi].
    """
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(1, 128)
        self.layer2 = Linear(128, 128)
        self.layer3 = Linear(128, 1)

    def forward(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        return self.layer3(x)  # no activation on the output layer

    def get_loss(self, x, y):
        return mse_loss(self.forward(x), y)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(3000):
            total_loss = 0.0
            count = 0
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
                count += len(x)
            if total_loss / count < 0.005:
                break


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.
    """
    def __init__(self):
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        self.layer1 = Linear(input_size, 256)
        self.layer2 = Linear(256, 128)
        self.layer3 = Linear(128, output_size)

    def run(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        return self.layer3(x)  # raw scores (logits), no ReLU on the last layer

    def forward(self, x):
        return self.run(x)

    def get_loss(self, x, y):
        return cross_entropy(self.run(x), y)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(30):
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
            if dataset.get_validation_accuracy() >= 0.98:
                break


class LanguageIDModel(Module):
    """
    Language identification at single-word granularity, using a simple RNN.
    """
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        self.hidden_size = 200
        self.w_x = Linear(self.num_chars, self.hidden_size)   # input -> hidden
        self.w_h = Linear(self.hidden_size, self.hidden_size)  # hidden -> hidden
        self.w_out = Linear(self.hidden_size, len(self.languages))

    def run(self, xs):
        # first character initialises the hidden state
        h = relu(self.w_x(xs[0]))
        # each later character updates it, carrying information forward
        for x in xs[1:]:
            h = relu(self.w_x(x) + self.w_h(h))
        return self.w_out(h)  # (batch_size x 5) logits

    def get_loss(self, xs, y):
        return cross_entropy(self.run(xs), y)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.002)
        for epoch in range(40):
            for batch in dataloader:
                xs = movedim(batch['x'], 0, 1)  # (L x batch x num_chars)
                y = batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(xs, y)
                loss.backward()
                optimizer.step()
            if dataset.get_validation_accuracy() >= 0.86:
                break


def Convolve(input: tensor, weight: tensor):
    """
    2D convolution (no padding) using only slicing and arithmetic.
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    out_h = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    out_w = input_tensor_dimensions[1] - weight_dimensions[1] + 1
    Output_Tensor = input[:out_h, :out_w] * 0
    # each weight value scales a shifted window of the input; summing them
    # gives the same result as sliding the kernel over the input
    for a in range(weight_dimensions[0]):
        for b in range(weight_dimensions[1]):
            Output_Tensor = Output_Tensor + weight[a, b] * input[a:a + out_h, b:b + out_w]
    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    Convolutional model for MNIST digit classification.
    """
    def __init__(self):
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        # a 3x3 kernel on a 28x28 image gives a 26x26 output, flattened to 676
        self.layer1 = Linear(26 * 26, 128)
        self.layer2 = Linear(128, output_size)

    def run(self, x):
        return self(x)

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        x = relu(self.layer1(x))
        return self.layer2(x)

    def get_loss(self, x, y):
        return cross_entropy(self(x), y)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(10):
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x, y)
                loss.backward()
                optimizer.step()
            if hasattr(dataset, 'get_validation_accuracy') and dataset.get_validation_accuracy() >= 0.9:
                break


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

        self.layer_size = layer_size

    def forward(self, input):
        B, T, C = input.size()

        K = self.k_layer(input)  # (B, T, C)
        Q = self.q_layer(input)  # (B, T, C)
        V = self.v_layer(input)  # (B, T, C)

        # scaled dot-product scores between every pair of positions: (B, T, T)
        M = matmul(K, Q.transpose(-2, -1)) / (self.layer_size ** 0.5)

        # causal mask: a position may not attend to later positions
        M = M.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))[0]

        # turn scores into attention weights over the last dimension
        M = softmax(M, dim=-1)

        # weighted sum of the value vectors: (B, T, C)
        return matmul(M, V)