from typing import Callable
import torch
from torch.nn.parameter import Parameter
from torch.nn import Module


class NegEntropyIndependence:
    """
    Мера независимости.
    Представляет собой обратное к сумме негэнтропий случайных величин, которые предварительно обелены.
    На вход подается вектор случайных величин, элементы которого -- выборки этих случайных величин.
    """

    def __init__(self, negentropy_approx: Callable, preprocessing: Callable, factor : float = 1.0):
        """
        :param negentropy_approx: Callable, некоторая функция по вычислению приближение негэнтропии
        :param preprocessing: Callable, функция по выполнению предобработки входных данных
        :param factor: float, то, с каким весом вычисляется метрика
        """
        self.negentropy = negentropy_approx
        self.preprocessing = preprocessing
        self.factor = factor

    def __call__(self, x: torch.Tensor) -> float:
        """
        Вычисление меры независимости.
        :param x: torch.Tensor, вектор случайных величин, элементы которого -- выборки этих случайных величин
        :return: float, результат вычисления меры независимости
        """
        device = x.device.type

        assert len(x.size()) == 2

        x_preproc = self.preprocessing(x)

        negentropy_for_each_variable = torch.zeros((x.size(0), 1)).to(device)
        for i in range(x.size(0)):
            negentropy_for_each_variable[i] = self.negentropy(x_preproc[i])
        # если получили отрицательные значения -- значит из-за погрешности ушли в отрицательные
        # поэтому нужно занулить их
        negentropy_for_each_variable = torch.nn.functional.relu(negentropy_for_each_variable)
        return torch.div(self.factor, torch.sum(negentropy_for_each_variable))


"""
    Далее -- предобработка вектора случайных величин(а именно -- обеление), элементы которого -- выборки этих случайных величин.
    Выполняется приближенный ZCA.
    Реализация взята:  https://github.com/cvlab-epfl/Power-Iteration-SVD/tree/master
"""


class PowerIterationOnce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = torch.eye(M.shape[-1], out=torch.empty_like(M))
        numerator = I - v_k.mm(torch.t(v_k))
        denominator = torch.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = torch.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak


class ZCANormSVDPI(Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True, device: str = 'cpu'):
        super(ZCANormSVDPI, self).__init__()
        self.device = device
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1).to(self.device))
        self.bias = Parameter(torch.Tensor(num_features, 1).to(self.device))
        self.power_layer = PowerIterationOnce.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1).to(self.device))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length).to(self.device))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1).to(self.device))

    def reset_running_stats(self):
        self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        # print(f"x shape in zca = {x}")
        x = x[:, :, None, None]
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + (torch.eye(C, out=torch.empty_like(x).to(self.device))) * self.eps

            assert C % G == 0
            length = int(C / G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0) / e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            # print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i]).to(self.device)
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij) / torch.mm(eigenvector_ij.t(),
                                                                                                    eigenvector_ij)
                    if lambda_ij < 0:
                        print('eigenvalues: ', e)
                        print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    diff_ratio = (lambda_ij - e[j]).abs() / e[j]
                    if diff_ratio > 0.1:
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return torch.transpose(torch.squeeze(xr), 0, 1)

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)

            return torch.transpose(torch.squeeze(x), 0, 1)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(ZCANormSVDPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
