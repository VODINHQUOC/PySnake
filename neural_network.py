import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, weights_ih=None, weights_ho=None, bias_h=None, bias_o=None):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        if weights_ih is None:
            self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * np.sqrt(2. / self.input_nodes) # He initialization
        else:
            self.weights_ih = weights_ih

        if weights_ho is None:
            self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(2. / self.hidden_nodes)
        else:
            self.weights_ho = weights_ho

        if bias_h is None:
            self.bias_h = np.random.randn(self.hidden_nodes, 1)
        else:
            self.bias_h = bias_h

        if bias_o is None:
            self.bias_o = np.random.randn(self.output_nodes, 1)
        else:
            self.bias_o = bias_o

    def feedforward(self, input_array):
        # Đảm bảo input là cột vector
        inputs = np.array(input_array, ndmin=2).T

        # Tính toán tín hiệu vào lớp ẩn
        hidden_inputs = np.dot(self.weights_ih, inputs) + self.bias_h
        # Tính toán tín hiệu ra khỏi lớp ẩn (qua hàm kích hoạt)
        hidden_outputs = sigmoid(hidden_inputs)

        # Tính toán tín hiệu vào lớp output
        final_inputs = np.dot(self.weights_ho, hidden_outputs) + self.bias_o
        # Tính toán tín hiệu ra khỏi lớp output (qua hàm kích hoạt)
        final_outputs = sigmoid(final_inputs) # Hoặc softmax nếu cần xác suất

        return final_outputs.flatten() # Trả về mảng 1 chiều

    def mutate(self, mutation_rate):
        # Đột biến trọng số IH
        mask = np.random.rand(*self.weights_ih.shape) < mutation_rate
        noise = np.random.randn(*self.weights_ih.shape) * 0.1 # Nhiễu nhỏ
        self.weights_ih += mask * noise

        # Đột biến trọng số HO
        mask = np.random.rand(*self.weights_ho.shape) < mutation_rate
        noise = np.random.randn(*self.weights_ho.shape) * 0.1
        self.weights_ho += mask * noise

        # Đột biến bias H
        mask = np.random.rand(*self.bias_h.shape) < mutation_rate
        noise = np.random.randn(*self.bias_h.shape) * 0.1
        self.bias_h += mask * noise

        # Đột biến bias O
        mask = np.random.rand(*self.bias_o.shape) < mutation_rate
        noise = np.random.randn(*self.bias_o.shape) * 0.1
        self.bias_o += mask * noise

    def crossover(self, partner):
        child_nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)

        # Lai ghép trọng số IH (ví dụ: single point crossover)
        mid_point = np.random.randint(0, self.weights_ih.size)
        flat_w_ih1 = self.weights_ih.flatten()
        flat_w_ih2 = partner.weights_ih.flatten()
        child_flat_w_ih = np.concatenate((flat_w_ih1[:mid_point], flat_w_ih2[mid_point:]))
        child_nn.weights_ih = child_flat_w_ih.reshape(self.weights_ih.shape)

        # Lai ghép trọng số HO
        mid_point = np.random.randint(0, self.weights_ho.size)
        flat_w_ho1 = self.weights_ho.flatten()
        flat_w_ho2 = partner.weights_ho.flatten()
        child_flat_w_ho = np.concatenate((flat_w_ho1[:mid_point], flat_w_ho2[mid_point:]))
        child_nn.weights_ho = child_flat_w_ho.reshape(self.weights_ho.shape)

        # Lai ghép bias H
        mid_point = np.random.randint(0, self.bias_h.size)
        flat_b_h1 = self.bias_h.flatten()
        flat_b_h2 = partner.bias_h.flatten()
        child_flat_b_h = np.concatenate((flat_b_h1[:mid_point], flat_b_h2[mid_point:]))
        child_nn.bias_h = child_flat_b_h.reshape(self.bias_h.shape)

        # Lai ghép bias O
        mid_point = np.random.randint(0, self.bias_o.size)
        flat_b_o1 = self.bias_o.flatten()
        flat_b_o2 = partner.bias_o.flatten()
        child_flat_b_o = np.concatenate((flat_b_o1[:mid_point], flat_b_o2[mid_point:]))
        child_nn.bias_o = child_flat_b_o.reshape(self.bias_o.shape)

        return child_nn

    def clone(self):
        # Tạo bản sao sâu của mạng nơ-ron
        cloned_nn = NeuralNetwork(
            self.input_nodes, self.hidden_nodes, self.output_nodes,
            weights_ih=self.weights_ih.copy(),
            weights_ho=self.weights_ho.copy(),
            bias_h=self.bias_h.copy(),
            bias_o=self.bias_o.copy()
        )
        return cloned_nn