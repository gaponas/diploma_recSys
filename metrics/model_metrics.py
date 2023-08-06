import torch
import numpy as np
from typing import Callable, Tuple
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error

from negentropy_approximations.approximations import NegentropyApprox5
from sparsity import Sparsity
from independence import NegEntropyIndependence, ZCANormSVDPI


class Metrics:
    """
    Класс со всеми метриками, которые используются для оценки моделей
    """
    def __init__(self, negentropy_approx: Callable = None):
        """
        :param negentropy_approx: Callable,какой приближение негэнтропии использовать при вычислении оценки независимости
        """
        if negentropy_approx is None:
            # по умолчанию будем использовать это приближение, т.к. в экспериментах он дал наилучшие результаты для размеров выборки >200
            negentropy_approx = NegentropyApprox5()
        self.negentropy_approx = negentropy_approx

    # -----точность--------

    def accurasy(self, expected: torch.Tensor, predicted: torch.Tensor, k_top: int = 10) -> Tuple[float, float]:
        """
        Функция по вычислению метрик "точности" предсказаний модели: RMSE, MSE, NDCG.
            Предполагается, что переданные данные относятся к 1 пользователю.
        :param expected: torch.Tensor, реальные значения :param predicted: torch.Tensor, значения, предсказанные
        моделью :param k_top: int, для какого размера топа вычисляем значение NDSG
        :return: Tuple[float, float], возвращает пару значений RMSE и NDCG для переданных данных
        """
        mse = self._MSE(expected, predicted)
        return np.sqrt(mse + 1e-6), mse, self._NDCG(expected, predicted, k_top)

    def _MSE(self, expected: torch.Tensor, predicted: torch.Tensor) -> float:
        device = expected.device.type
        if device != 'cpu':
            expected = expected.to('cpu')
            predicted = predicted.to('cpu')
        return mean_squared_error(expected, predicted)

    def _MAE(self, expected: torch.Tensor, predicted: torch.Tensor) -> float:
        device = expected.device.type
        if device != 'cpu':
            expected = expected.to('cpu')
            predicted = predicted.to('cpu')
        return mean_absolute_error(expected, predicted)

    def _NDCG(self, expected: torch.Tensor, predicted: torch.Tensor, k_top: int = 10) -> float:
        device = expected.device.type
        if device != 'cpu':
            expected = expected.to('cpu')
            predicted = predicted.to('cpu')

        expected = torch.squeeze(expected)
        predicted = torch.squeeze(predicted)

        return ndcg_score([expected.numpy()], [predicted.numpy()], k=k_top)

    # ------разреженность------

    def sparsity(self, batch_of_vectors: torch.Tensor, decimals: int = 1) -> float:
        """
        Функция по вычислению метрики разреженности скрытых векторных представлений.
        :param batch_of_vectors: torch.Tensor, набор векторных представлений
        :param decimals: int, количество значащих знаков при вычислении разреженности
        :return: float, значение меры разреженности для переданных данных
        """
        device = batch_of_vectors.device.type
        if device != 'cpu':
            batch_of_vectors = batch_of_vectors.to('cpu')

        return Sparsity(decimals=decimals)(batch_of_vectors)

    # ------независимость--------

    def independence(self, batch_of_vectors: torch.Tensor, decimals: int = 2):
        """
        Функция по вычислению метрик независимости компонент скрытых векторных представлений:
            через приближение взаимной информации и через разреженность матрицы корреляций(вне диагонали)
        :param batch_of_vectors: torch.Tensor, набор векторных представлений
        :param decimals: int, количество значащих знаков при вычислении разреженности матрицы корреляций
        """
        return self._correlation_matrix(batch_of_vectors, decimals), self._negentropy_diff(batch_of_vectors)

    def _correlation_matrix(self, batch_of_vectors, decimals: int = 2):
        device = batch_of_vectors.device.type
        if device != 'cpu':
            batch_of_vectors = batch_of_vectors.to('cpu')

        batch_of_vectors = torch.transpose(batch_of_vectors, 0, 1)

        count_of_non_diag_els = batch_of_vectors.shape[0] * (batch_of_vectors.shape[0] - 1)
        corr_matr = torch.corrcoef(batch_of_vectors)

        nan_values = torch.isnan(corr_matr)
        corr_matr[nan_values] = 0.0

        non_diag_corr_matr = torch.abs(corr_matr).fill_diagonal_(0.0)
        rounded = torch.round(non_diag_corr_matr, decimals=decimals)
        not_null = torch.where(rounded > 0, 1, 0)
        return torch.sum(not_null) / count_of_non_diag_els

    def _negentropy_diff(self, batch_of_vectors):
        device = batch_of_vectors.device.type
        if device != 'cpu':
            batch_of_vectors = batch_of_vectors.to('cpu')

        batch_of_vectors = torch.transpose(batch_of_vectors, 0, 1)

        neg_sum_f = NegEntropyIndependence(self.negentropy_approx, ZCANormSVDPI(batch_of_vectors.shape[0]))
        neg_sum = neg_sum_f(batch_of_vectors)
        return neg_sum
