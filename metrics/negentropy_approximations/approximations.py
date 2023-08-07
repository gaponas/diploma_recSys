"""
    Файл с приближениями негэнтропии и вспомогательными объектами
"""

from typing import Callable
from scipy.stats import norm
import torch
import numpy as np

"""
    Далее -- приближения из статьи https://www.cs.jhu.edu/~ayuille/courses/Stat161-261-Spring14/HyvO00-icatut.pdf
"""


class NormalMeanG:
    """
    вычисление E(G(v)), где G -- некоторая функция, v -- нормальная величина
    """

    def __init__(self, G: Callable, loc: float = 0, scale: float = 1):
        """
        Задаем параметры для вычисления значения
        :param G: Callable, функция, которую применяем к нормальной величине и вычисляем среднее
        :param loc: float, среднее нормальной величины v
        :param scale: float, среднеквадратичное отклонение нормальной величины v
        """
        self.g = G
        normal_vect = norm.rvs(loc=loc, scale=scale, size=100000)
        normal_tensor = torch.from_numpy(normal_vect)
        normal_g = self.g(normal_tensor).numpy()
        self.mean = normal_g.mean()

    def __call__(self) -> float:
        """
        Возвращает значение E(G(v)) с заданными при инициализации параметрами
        :return: float, значение E(G(v))
        """
        return self.mean


class G1:
    """
    Заданная в формуле 26 функция G1 для вычисления приближения
    """

    def __init__(self, a_const: float):
        """
        :param a_const: float, константа для вычисления функции, должна быть задана в диапазоне от 1 до 2
        """
        assert (1 <= a_const <= 2)
        self.a = a_const

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """
        Возвращает результат применения функции к каждому элементу переданной выборки
        :param u: torch.Tensor, выборка из некоторой случайной величины
        :return: torch.Tensor, результат применения функции к выборке
        """
        return torch.mul(1 / self.a, torch.log(torch.cosh(torch.mul(u, self.a))))


class G2:
    """
    Заданная в формуле 26 функция G2 для вычисления приближения
    """

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """
        Возвращает результат применения функции к каждому элементу переданной выборки
        :param u: torch.Tensor, выборка из некоторой случайной величины
        :return: torch.Tensor, результат применения функции к выборке
        """
        return torch.mul(-1, torch.exp(torch.div(torch.pow(u, 2), (-2))))


class NegentropyApprox1:
    """
    Приближение негэнтропии из формулы 25 с использованием G1
    """

    def __init__(self, a_const: float):
        """
        :param a_const: float, константа из функции G1
        """
        self.g = G1(a_const)
        self.normal_g_mean = NormalMeanG(self.g)()

    def __call__(self, x: torch.Tensor) -> float:
        """
        Возвращает значение приближения негэнтропии для переданной выборки
        :param x: torch.Tensor, выборка, для которой нужно вычислить негэнтропию
        :return: float, значение приближение негэнтропии для переданной выборки
        """
        g_x = self.g(x)
        return torch.pow(torch.sub(torch.mean(g_x), self.normal_g_mean), 2)


class NegentropyApprox2:
    """
    Приближение негэнтропии из формулы 25 с использованием G2
    """

    def __init__(self):
        self.g = G2()
        self.normal_g_mean = NormalMeanG(self.g)()

    def __call__(self, x: torch.Tensor) -> float:
        """
        Возвращает значение приближения негэнтропии для переданной выборки
        :param x: torch.Tensor, выборка, для которой нужно вычислить негэнтропию
        :return: float, значение приближение негэнтропии для переданной выборки
        """
        g_x = self.g(x)
        return torch.pow(torch.sub(torch.mean(g_x), self.normal_g_mean), 2)


class NegentropyApprox3:
    """
    Приближение негэнтропии из формулы 23
    """

    def __call__(self, x: torch.Tensor) -> float:
        """
        Возвращает значение приближения негэнтропии для переданной выборки
        :param x: torch.Tensor, выборка, для которой нужно вычислить негэнтропию
        :return: float, значение приближение негэнтропии для переданной выборки
        """
        x_3 = torch.pow(x, 3)
        term1 = torch.mul(1 / 12, torch.pow(torch.mean(x_3), 2))
        term2 = torch.mul(1 / 48, torch.pow(self.__kurt(x), 2))
        return term1 + term2

    def __kurt(self, x: torch.Tensor) -> float:
        x_2 = torch.pow(x, 2)
        x_4 = torch.pow(x, 4)
        term1 = torch.mean(x_4)
        term2 = torch.mul(torch.pow(torch.mean(x_2), 2), 3)
        return torch.sub(term1, term2)


"""
Далее -- приближения из статьи https://www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf
"""


class NegentropyApprox4:
    """
    Приближение негэнтропии из формулы 8
    """
    def __init__(self):
        self.k1 = 36 / (8 * np.sqrt(3) - 9)
        self.k2 = 1 / (2 - 6 / np.pi)
        self.coef = np.sqrt(2 / np.pi)
        self.g2 = G2()

    def __call__(self, x: torch.Tensor) -> float:
        """
        Возвращает значение приближения негэнтропии для переданной выборки
        :param x: torch.Tensor, выборка, для которой нужно вычислить негэнтропию
        :return: float, значение приближение негэнтропии для переданной выборки
        """
        term1 = torch.mul(self.k1, torch.pow(torch.mean(torch.mul(-x, self.g2(x))), 2))
        term2 = torch.mul(self.k2, torch.pow(torch.sub(torch.mean(torch.abs(x)), self.coef), 2))
        return term1 + term2


class NegentropyApprox5:
    """
        Приближение негэнтропии из формулы 9
        """
    def __init__(self):
        self.k1 = 36 / (8 * np.sqrt(3) - 9)
        self.k2 = 24 / (16 * np.sqrt(3) - 27)
        self.coef = np.sqrt(1 / 2)
        self.g2 = G2()

    def __call__(self, x: np.ndarray) -> float:
        """
        Возвращает значение приближения негэнтропии для переданной выборки
        :param x: torch.Tensor, выборка, для которой нужно вычислить негэнтропию
        :return: float, значение приближение негэнтропии для переданной выборки
        """
        term1 = torch.mul(self.k1, torch.pow(torch.mean(torch.mul(-x, self.g2(x))), 2))
        term2 = torch.mul(self.k2, torch.pow(torch.sub(torch.mean(-self.g2(x)), self.coef), 2))
        return term1 + term2