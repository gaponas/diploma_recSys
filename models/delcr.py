from torch.nn import Module, Linear, Tanh, ReLU, Sequential, MSELoss
from typing import List, Callable
from itertools import chain
import torch
from metrics.independence import NegEntropyIndependence, ZCANormSVDPI
from metrics.sparsity import Sparsity


class DELCR(Module):
    """
    Модель DELCR, написанная на основании статьи https://www.researchgate.net/publication/366885365_Deep_Learning_and_Embedding_Based_Latent_Factor_Model_for_Collaborative_Recommender_Systems
    """

    def __init__(self, user_input_size: int, item_input_size: int, embedding_size: int, latent_factor_size: int,
                 hidden_outputs: List[int]):
        """
        :param user_input_size: int, размер входного вектора о пользователе
        :param item_input_size: int, размер входного вектора об элементе
        :param embedding_size: int, размер плотного вектора элементов, по которому выполняется начальное предсказание
        :param latent_factor_size: int, размеры внутренних слоев модели
        :param hidden_outputs: List[int], размер скрытого векторного представления сущностей
        """
        super(DELCR, self).__init__()
        self.user_embedding = self._embedding_layer(user_input_size, embedding_size)
        self.item_embedding = self._embedding_layer(item_input_size, embedding_size)

        self.user_nn = self._nn_layer(embedding_size, latent_factor_size, hidden_outputs)
        self.item_nn = self._nn_layer(embedding_size, latent_factor_size, hidden_outputs)

    def _embedding_layer(self, input_size, emdedding_size):
        return Linear(input_size, emdedding_size, bias=False)

    def _nn_layer(self, input_size, output_size, hidden_outputs):
        dims = [input_size] + hidden_outputs
        layers_count = len(dims) - 1
        layers = list(chain.from_iterable(
            (Linear(dims[i], dims[i + 1], bias=True), Tanh()) for i in range(layers_count)
        ))
        layers.append(Linear(dims[-1], output_size))
        layers.append(ReLU())
        return Sequential(*layers)

    def _row_wise_dotprod(self, a, b):
        num_rows, num_cols = a.size(0), a.size(1)
        prod = torch.bmm(torch.reshape(a, (num_rows, 1, num_cols)), torch.reshape(b, (num_rows, num_cols, 1)))
        return prod

    def forward(self, users, items):
        """
        На выходе для каждой пары пользователь-элемент:
            предсказание на основании скрытых векторных представлений; предсказание на основании плотных векторов;
            скрытое представление пользователя; скрытое представление элемента
        """
        user_emb = self.user_embedding(users.float())
        item_emb = self.item_embedding(items.float())
        emb_dot = self._row_wise_dotprod(user_emb, item_emb)

        user_latent_factor = self.user_nn(user_emb)
        item_latent_factor = self.item_nn(item_emb)
        result = self._row_wise_dotprod(user_latent_factor, item_latent_factor)
        # предсказание на выходе, предсказание на уровне эмбеддингов
        return result, emb_dot, user_latent_factor, item_latent_factor

    def get_user_embeddings(self, users):
        user_emb = self.user_embedding(users.float())
        user_latent_factor = self.user_nn(user_emb)
        return user_latent_factor

    def get_item_embeddings(self, items):
        item_emb = self.item_embedding(items.float())
        item_latent_factor = self.item_nn(item_emb)
        return item_latent_factor


class DelsrLoss(Module):
    """
    Оригинальная функция потерь для DELCR модели
    """
    def __init__(self):
        super(DelsrLoss, self).__init__()
        self.loss1 = MSELoss()
        self.loss2 = MSELoss()

    def forward(self, real_rating, final_rating, embed_rating):
        loss_accuracy = self.loss1(embed_rating, real_rating) + self.loss2(final_rating, real_rating)
        loss = loss_accuracy
        if torch.isnan(torch.sum(final_rating)) or torch.isnan(torch.sum(embed_rating)):
            return
        return loss


class IndepDelsrLoss(Module):
    """
    Функция потерь для модели DELCR с добавлением ограничения на независимость компонент векторных представлений
    """
    def __init__(self, negentropy_approx1: Callable, negentropy_approx2: Callable, lf_size: int, device: str = 'cpu',
                 indep_w: int = 1.0):
        """
        :param negentropy_approx1: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений пользователей
        :param negentropy_approx2: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений элементов
        :param lf_size: int, размер построенного моделью скрытого представления
        :param device: str, на чем выполняются вычисления
        :param indep_w: float, вес независимости в функции потерь
        """
        super(IndepDelsrLoss, self).__init__()
        self.loss1 = MSELoss()
        self.loss2 = MSELoss()
        self.indep_loss1 = NegEntropyIndependence(negentropy_approx1, ZCANormSVDPI(lf_size, device=device), factor=indep_w)
        self.indep_loss2 = NegEntropyIndependence(negentropy_approx2, ZCANormSVDPI(lf_size, device=device), factor=indep_w)

    def forward(self, real_rating, final_rating, embed_rating, user_lf, item_lf):
        loss_accuracy = self.loss1(embed_rating, real_rating) + self.loss2(final_rating, real_rating)
        loss_indep = self.indep_loss1(user_lf.t()) + self.indep_loss2(item_lf.t())
        loss = loss_accuracy + loss_indep
        if torch.isnan(torch.sum(final_rating)) or torch.isnan(torch.sum(embed_rating)) or torch.isnan(
                torch.sum(user_lf)) or torch.isnan(torch.sum(item_lf)):
            return
        return loss


class IndepSparseDelcrLoss(Module):
    """
    Функция потерь для модели DELCR с добавлением ограничения на независимость компонент и разреженность векторных представлений
    """
    def __init__(self, negentropy_approx1: Callable, negentropy_approx2: Callable, lf_size: int, device: str = 'cpu',
                 indep_w: float = 1.0, sparse_w: float = 1.0):
        """
        :param negentropy_approx1: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений пользователей
        :param negentropy_approx2: Callable, функция для вычисления приближения негэнтропии,
          которую необходимо использовать при вычислении меры независимости компонент представлений элементов
        :param lf_size: int, размер построенного моделью скрытого представления
        :param device: str, на чем выполняются вычисления
        :param indep_w: float, вес независимости в функции потерь
        :param sparse_w: float, вес разреженности в функции потерь
        """
        super(IndepSparseDelcrLoss, self).__init__()
        self.loss1 = MSELoss()
        self.loss2 = MSELoss()
        self.indep_loss1 = NegEntropyIndependence(negentropy_approx1, ZCANormSVDPI(lf_size, device=device), factor=indep_w)
        self.indep_loss2 = NegEntropyIndependence(negentropy_approx2, ZCANormSVDPI(lf_size, device=device), factor=indep_w)
        self.sparse_loss1 = Sparsity(sparse_w)
        self.sparse_loss2 = Sparsity(sparse_w)

    def forward(self, real_rating, final_rating, embed_rating, user_lf, item_lf):
        loss_accuracy = self.loss1(embed_rating, real_rating) + self.loss2(final_rating, real_rating)
        loss_indep = self.indep_loss1(user_lf.t()) + self.indep_loss2(item_lf.t())
        loss_sparse = self.sparse_loss1(user_lf) + self.sparse_loss2(item_lf)
        loss = loss_accuracy + loss_indep + loss_sparse
        if torch.isnan(torch.sum(final_rating)) or torch.isnan(torch.sum(embed_rating)) or torch.isnan(
                torch.sum(user_lf)) or torch.isnan(torch.sum(item_lf)):
            return
        return loss
