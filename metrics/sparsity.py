import torch


class Sparsity:
    """
    Мера разреженности векторов.
    """
    def __init__(self, factor: float = 1.0, decimals: int = 1):
        """
        :param factor: float, то, с каким весом вычисляется метрика
        :param decimals: int, количество знаков после запятой, которые считаются значащими
        """
        self.factor = factor
        self.decimals = decimals

    def __call__(self, x: torch.Tensor):
        """
        Вычисление меры разреженности. Разреженностью считается средняя разреженность по всем векторам
        :param x: torch.Tensor, набор векторов.
        :return: результат вычисления меры разреженности
        """
        rounded = torch.round(x, decimals=self.decimals)
        is_null = torch.where(rounded > 0, 1, 0).float()
        sparcity_along_axis = torch.mean(is_null, dim=1)
        return torch.mul(torch.mean(sparcity_along_axis), self.factor)
