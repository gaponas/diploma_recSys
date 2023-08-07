from typing import List, Callable
from torch.nn import Module, Linear, ReLU, Sequential
from itertools import chain
from torch.nn.functional import cosine_similarity
import torch
from metrics.independence import NegEntropyIndependence, ZCANormSVDPI
from metrics.sparsity import Sparsity


class DMF(Module):
    """
    Модель deepMF, написанная на основании статьи https://www.ijcai.org/Proceedings/2017/0447.pdf
    """
    def __init__(self, user_input_size: int, item_input_size: int, latent_factor_size: int,
                 hidden_outputs: List[int]):
        """
        :param user_input_size: int, размер входного вектора о пользователе
        :param item_input_size: int, размер входного вектора об элементе
        :param latent_factor_size: int, размеры внутренних слоев модели
        :param hidden_outputs: List[int], размер скрытого векторного представления сущностей
        """
        super(DMF, self).__init__()

        self.user_layers = self._nn_layer(user_input_size, hidden_outputs, latent_factor_size)
        self.item_layers = self._nn_layer(item_input_size, hidden_outputs, latent_factor_size)
        self.cosine = cosine_similarity

    def _nn_layer(self, input_size, hidden_outputs, output_size):
        dims = [input_size] + hidden_outputs + [output_size]
        layers_count = len(dims) - 1
        layers = list(chain.from_iterable(
            (ReLU(), Linear(dims[i], dims[i + 1], bias=True)) for i in range(layers_count)
        ))
        layers.append(ReLU())
        return Sequential(*layers)

    def forward(self, users, items):
        """
        На выходе для каждой пары пользователь-элемент:
          предсказание рейтинга, скрытое представление пользователя, скрытое представление элемента.
        """
        device = users.device.type

        users_hidden = self.user_layers(users.float())
        items_hidden = self.item_layers(items.float())
        result = self.cosine(users_hidden, items_hidden)
        result = torch.maximum(result, torch.full(result.shape, 1e-6).to(device))

        return result, users_hidden, items_hidden


class DMFLoss(Module):
    """
    Оригинальная функция потерь для deepMF модели
    """

    def __init__(self, max_rate: float = 5.0):
        """
        :param max_rate: float, верхняя граница предсказываемых значений
        """
        super(DMFLoss, self).__init__()
        self.R = max_rate

    def forward(self, real_r, pred_r):
        term1 = torch.div(torch.mul(torch.log(pred_r), real_r), self.R)
        term2 = torch.mul(torch.sub(1, torch.div(real_r, self.R)), torch.log(torch.sub(1, pred_r)))

        loss = term1 + term2
        return torch.mul(-1, torch.sum(loss))


class IndepDMFLoss(Module):
    """
    Функция потерь для модели DeepMF с добавлением ограничения на независимость компонент векторных представлений
    """

    def __init__(self, negentropy_approx1: Callable, negentropy_approx2: Callable, lf_size: int, device: str = 'cpu',
                 indep_w: float = 1.0, max_rate: float = 5.0):
        """
        :param negentropy_approx1: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений пользователей
        :param negentropy_approx2: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений элементов
        :param lf_size: int, размер построенного моделью скрытого представления
        :param device: str, на чем выполняются вычисления
        :param indep_w: float, вес независимости в функции потерь
        :param max_rate: float, верхняя граница предсказываемых значений
        """
        super(IndepDMFLoss, self).__init__()
        self.R = max_rate
        self.indep_loss1 = NegEntropyIndependence(negentropy_approx1, ZCANormSVDPI(lf_size, device=device),
                                                  factor=indep_w)
        self.indep_loss2 = NegEntropyIndependence(negentropy_approx2, ZCANormSVDPI(lf_size, device=device),
                                                  factor=indep_w)

    def forward(self, real_r, pred_r, user_lf, item_lf):
        term1 = torch.div(torch.mul(torch.log(pred_r), real_r), self.R)
        term2 = torch.mul(torch.sub(1, torch.div(real_r, self.R)), torch.log(torch.sub(1, pred_r)))

        terms_sum = term1 + term2
        loss_cos = torch.mul(-1, torch.sum(terms_sum))
        loss_indep = self.indep_loss1(user_lf.t()) + self.indep_loss2(item_lf.t())
        print(loss_cos, loss_indep)
        loss = loss_cos + loss_indep
        return loss


class IndepSparseDMFLoss(Module):
    """
    Функция потерь для модели DeepMF с добавлением ограничения на независимость компонент и разреженность векторных представлений
    """
    def __init__(self, negentropy_approx1: Callable, negentropy_approx2: Callable, lf_size: int, device: str = 'cpu',
                 indep_w: float = 1.0, sparse_w: float = 1.0, max_rate: float = 5.0):
        """
        :param negentropy_approx1: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений пользователей
        :param negentropy_approx2: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений элементов
        :param lf_size: int, размер построенного моделью скрытого представления
        :param device: str, на чем выполняются вычисления
        :param indep_w: float, вес независимости в функции потерь
        :param sparse_w: float, вес разреженности в функции потерь
        :param max_rate: float, верхняя граница предсказываемых значений
        """
        super(IndepDMFLoss, self).__init__()
        self.R = max_rate
        self.indep_loss1 = NegEntropyIndependence(negentropy_approx1, ZCANormSVDPI(lf_size, device=device),
                                                  factor=indep_w)
        self.indep_loss2 = NegEntropyIndependence(negentropy_approx2, ZCANormSVDPI(lf_size, device=device),
                                                  factor=indep_w)
        self.sparse_loss1 = Sparsity(sparse_w)
        self.sparse_loss2 = Sparsity(sparse_w)

    def forward(self, real_r, pred_r, user_lf, item_lf):
        term1 = torch.div(torch.mul(torch.log(pred_r), real_r), self.R)
        term2 = torch.mul(torch.sub(1, torch.div(real_r, self.R)), torch.log(torch.sub(1, pred_r)))

        terms_sum = term1 + term2
        loss_cos = torch.mul(-1, torch.sum(terms_sum))
        loss_indep = self.indep_loss1(user_lf.t()) + self.indep_loss2(item_lf.t())
        loss_sparse = self.sparse_loss1(user_lf) + self.sparse_loss2(item_lf)
        print(loss_cos, loss_indep, loss_sparse)
        loss = loss_cos + loss_indep + loss_sparse
        return loss
