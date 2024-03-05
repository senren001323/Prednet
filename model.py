import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hard_sigmoid(x):
    '''
    Returns
    0 if x < -2.5,
    1 if x > 2.5.
    In -2.5 <= x <= 2.5, returns 0.2 * x + 0.5.
    '''
    slope = 0.2
    shift = 0.5
    x = (slope * x) + shift
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x

class PredNet(torch.nn.Module):
    def __init__(self,
                 A_Ahat_out_channels,
                 R_out_channels,
                 device):
        super(PredNet, self).__init__()
        self.A_Ahat_out_channels = A_Ahat_out_channels
        self.num_layers = len(A_Ahat_out_channels)
        self.R_out_channels = R_out_channels
        self.device = device

        self.make_layers()

    def isNotTopestLayer(self, layerIndex):
        '''judge if the layerIndex is not the topest layer.'''
        if layerIndex < self.num_layers - 1:
            return True
        else:
            return False

    def make_layers(self):
        self.conv_layers = {item: [] for item in ['i', 'f', 'c', 'o', 'A', 'Ahat']}
        lstm_list = ['i', 'f', 'c', 'o']

        for item in sorted(self.conv_layers.keys()):
            for layer in range(self.num_layers):
                if item == 'Ahat':
                    in_channels = self.R_out_channels[layer]  # R --> Ahat
                    self.conv_layers['Ahat'].append(torch.nn.Conv2d(in_channels=in_channels,
                                                                    out_channels=self.A_Ahat_out_channels[layer],
                                                                    kernel_size=3,
                                                                    stride=(1, 1),
                                                                    padding=1
                                                                    ))
                    self.conv_layers['Ahat'].append(torch.nn.ReLU())

                elif item == 'A':
                    if self.isNotTopestLayer(layer):  # A layers' length is 1 shorter then Ahat and R
                        in_channels = self.R_out_channels[layer] * 2   # E:[relu(Ahat-A), relu(A-Ahat)] --> A
                        self.conv_layers['A'].append(torch.nn.Conv2d(in_channels=in_channels,
                                                                     out_channels=self.A_Ahat_out_channels[layer + 1],
                                                                     kernel_size=3,
                                                                     stride=(1, 1),
                                                                     padding=1
                                                                     ))
                        self.conv_layers['A'].append(torch.nn.ReLU())

                elif item in lstm_list:  # building R units
                    in_channels = self.A_Ahat_out_channels[layer] * 2 + self.R_out_channels[layer]# layer_l: E=[Relu(A-Ahat);Relu(Ahat-A)] --> layer_l+1: A
                    #in_channels = self.R_out_channels[layer]
                    if self.isNotTopestLayer(layer):
                        in_channels += self.R_out_channels[layer + 1]
                    self.conv_layers[item].append(torch.nn.Conv2d(in_channels=in_channels,
                                                                  out_channels=self.R_out_channels[layer],
                                                                  kernel_size=3,
                                                                  stride=(1, 1),
                                                                  padding=1
                                                                  ))

                    for name, layerList in self.conv_layers.items():
                        self.conv_layers[name] = torch.nn.ModuleList(layerList)
                        setattr(self, name, self.conv_layers[name])

                    self.upSample = torch.nn.Upsample(scale_factor=2, mode='nearest')
                    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                    self.fc1 = torch.nn.Linear(1536, 128) # 1536 = 4x4x96
                    self.fc2 = torch.nn.Linear(128, 10)

    def step(self, A, states):
        '''
        Args:
            A: 4D tensor with the shape of (batch_size, 3, Height, Width).
            states: states' format is same with 'initial_states' in forward() implement,
            but the content is different in each training epoch, it will be replaced by E_list from the before step()
        '''
        n = self.num_layers
        R_current = states[:n]
        c_current = states[n:2*n]
        E_current = states[2*n:3*n]

        R_list = []
        c_list = []
        E_list = []

        # Update R units starting from the top.
        for layer in reversed(range(self.num_layers)):
            inputs = [R_current[layer], E_current[layer]]
            if self.isNotTopestLayer(layer):
                inputs.append(R_up)
            inputs = torch.cat(inputs, dim=1)

            in_gate = hard_sigmoid(self.conv_layers['i'][layer](inputs))
            forget_gate = hard_sigmoid(self.conv_layers['f'][layer](inputs))
            cell_gate = F.tanh(self.conv_layers['c'][layer](inputs))
            out_gate = hard_sigmoid(self.conv_layers['o'][layer](inputs))

            c_next = forget_gate * c_current[layer] + in_gate * cell_gate
            R_next = out_gate * F.tanh(c_next)

            if self.isNotTopestLayer(layer):
                pass
            else:
                R_top = R_next # Output from topest R
                output = self.pool(R_top)
                output = torch.flatten(output, start_dim=1) # (N, flatten_dim)
                output = self.fc1(output)
                output = self.fc2(output)

            c_list.insert(0, c_next)
            R_list.insert(0, R_next) # inversely insert

            if layer > 0:
                R_up = self.upSample(R_next).data

        # Update feedforward path starting from the bottom.
        for layer in range(self.num_layers):
            Ahat = self.conv_layers['Ahat'][2 * layer](R_list[layer])  # Ahat_l = ReLu(CONV(R_l)) in each layer
            Ahat = self.conv_layers['Ahat'][2 * layer + 1](Ahat)  # relu activate.
            if layer == 0:
                Ahat = torch.where(Ahat > 1, torch.ones_like(Ahat), Ahat) # passed through a saturating non-linearity set at the maximum pixel value
                frame_prediction = Ahat

            #print(A.size(), Ahat.size())
            E_up = F.relu(Ahat - A)
            E_down = F.relu(A - Ahat)

            E_list.append(torch.cat((E_up, E_down), dim=1)) # (N, A_Ahat_out_channels[layer]*2, Height/{2^layer}, weight/{2^layer})

            if self.isNotTopestLayer(layer): # Topest layer does not need to calculate A
                A = self.conv_layers['A'][2 * layer](E_list[layer])
                A = self.conv_layers['A'][2 * layer + 1](A)
                A = self.pool(A)  # target for next layer

        states = R_list + c_list + E_list
        return output, states, E_list, frame_prediction

    def forward(self, A0_time, initial_states):
        """
        A_0 is the input from dateloder. It's shape is [batch_size, timesteps, 3, Height, Weight]
        Initial_states is a list of pytorch tensors
        """
        A0_time = A0_time.transpose(0, 1)
        num_timesteps = A0_time.size()[0]
        states = initial_states
        output_list = []
        frame_list = []

        for t in range(num_timesteps):
            A_0 = A0_time[t, ...]
            #print(A_0.size())

            output, states, E_list, frame_prediction = self.step(A_0, states)
            for layer in range(self.num_layers):
                layer_error = torch.mean(torch.flatten(E_list[layer], start_dim=1), dim=-1, keepdim=True)
                all_error = layer_error if layer == 0 else torch.cat((all_error, layer_error), dim=-1) # (batch_size, num_layers)
            output_list.append(all_error)
            frame_list.append(frame_prediction)

        return output, output_list, frame_list # return last output from the topest R


    def init_states(self, input_shape):
        '''
        input_shape is (batch_size, timesteps, 3, Height, Width)
        Return:
         -initial list of 'R', 'c', 'E' units
        '''
        init_height = input_shape[3]
        init_width = input_shape[4]

        base_init_state = np.zeros(input_shape)
        for _ in range(2):
            base_init_state = np.sum(base_init_state, axis=-1) # (batch_size, timesteps, 3)
        base_init_state = np.sum(base_init_state, axis=1) # (batch_size, 3)

        initial_states = []
        initial_states_item = ['R', 'c', 'E']
        for item in initial_states_item:
            for layer in range(self.num_layers):
                downSample_factor = 2 ** layer
                row = init_height // downSample_factor
                col = init_width // downSample_factor
                if item in ['R', 'c']:
                    stack_size = self.R_out_channels[layer]
                elif item == 'E':
                    stack_size = self.A_Ahat_out_channels[layer] * 2
                output_size = stack_size * row * col # flattened size
                reducer = np.zeros((input_shape[2], output_size)) # (3, output_size)
                initial_state = np.dot(base_init_state, reducer) # (batch_size, output_size）

                output_shape = (-1, stack_size, row, col)
                initial_state = torch.FloatTensor(np.reshape(initial_state, output_shape)).to(self.device)
                initial_states += [initial_state]
        return initial_states
