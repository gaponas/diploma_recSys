import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple


class UserItemRatingDataset(Dataset):
    '''
    Класс для формирования Dataset, для дальнейшего использования при обучении модели,
    состоит только из userId, itemId, rating для пары userId-itemId (не использует дополнительную информацию)
    '''

    def __init__(self, usersId: List[int], itemsId: List[int], rating: List[float], interaction_matrix: pd.DataFrame):
        """
        :param List[int] usersId: набор Id пользователей
        :param itemsId: List[int], набор Id элементов рекомендаций
        :param rating: List[float], набор рейтингов, где rating[i,j] -- это оценка поставленная usersId[i] элементу itemsId[j]
        :param interaction_matrix: pd.DataFrame, матрица интеракций, которая используется в задаче.
          Передаем на вход, т.к. некоторые пользователи/элементы могут не иметь взаимодействий ни с кем,
          а в матрицу их включить необходимо.
        """
        self.user_tensor = torch.LongTensor(usersId)
        self.item_tensor = torch.LongTensor(itemsId)
        self.rating_tensor = torch.FloatTensor(rating)
        self.interations = interaction_matrix

    def __getitem__(self, idx):
        """
        :return: Пятерку [id пользователя, id элемента, рейтинг для пары, вектор пользователя, вектор элемента]
        """
        user = self.user_tensor[idx]
        item = self.item_tensor[idx]
        rating = self.rating_tensor[idx]
        user_vec = torch.squeeze(torch.tensor(self.interations.loc[[user]].values))
        col = item.numpy()
        item_vec = torch.squeeze(torch.tensor(self.interations[col].values))
        return user, item, rating, user_vec, item_vec

    def __len__(self):
        return self.user_tensor.size(0)


class DataPreprocessing:
    '''
    Класс для предобработки всех датасетов для задачи КФ РС с явным фидбеком, без вспомогательных данных.
    '''

    def __init__(self, user_col: str, item_col: str, rating_col: str, sorting_col: str,
                 min_rating: float, max_rating: float):
        """
        Задаем основные характеристики датасета: названия нужных в задаче столбцов
        :param user_col: str, имя столбца с id пользователя
        :param item_col: str, имя столбца с id рекомендуемого элемента
        :param sorting_col: str, имя столбца, который отвечает за время(по которому можно определить порядок оценок)
        :param rating_col: str, имя столбца с рейтингом, который поставил пользователь рекомендуемому элементу
        :param min_rating: float, минимально допустимый рейтинг
        :param max_rating: float -- максимально допустимый рейтинг
        """

        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.normilized_col = rating_col + "_normilized"
        self.sorting_col = sorting_col
        self.sorted_actions_col = "user_actions_sorted"
        self.min_rating = min_rating
        self.max_rating = max_rating

    def normilize_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
          Нормализация столбца рейтингов MinMax scaler, перенос всех значений в диапазон [0,1].
          :param df: pd.DataFrame, таблица в которой нужно нормализовать рейтинг
          :return: pd.DataFrame, таблица, в которой столбец с рейтингом заменен столбцом с нормализованным
        """
        res_df = df.copy()
        res_df[self.normilized_col] = (res_df[self.rating_col] - self.min_rating) / (self.max_rating - self.min_rating)
        return res_df.drop(columns=[self.rating_col])

    def original_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
          Операция, обратная нормализации.
          :param df: pd.DataFrame, таблица, в которой нужно восстановить оригинальные рейтинги из нормированных
          :return: pd.DataFrame, таблица, в которой столбец с нормализованным рейтингом заменен столбцом с оригинальным
        """
        res_df = df.copy()
        res_df[self.rating_col] = res_df[self.normilized_col] * (self.max_rating - self.min_rating) + self.min_rating
        return res_df.drop(columns=[self.normilized_col])

    def interaction_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
          Построение матрицы интеракций пользователи-элементы.
          :param df: pd.DataFrame, таблица с данными для построения матрицы интеракций
          :return: pd.DataFrame, матрица интеракций
        """
        if self.normilized_col in df.columns:
            ratings = self.normilized_col
        else:
            ratings = self.rating_col
        return df.pivot_table(values=ratings, columns=self.item_col, index=self.user_col).fillna(0)

    def _remove_ratings_from_interaction_matrix(self, interactions: pd.DataFrame, df: pd.DataFrame):
        """
        Функция для удаления рейтингов из таблицы интеракций
        :param interactions: pd.DataFrame, матрица интеракций
        :param df: pd.DataFrame, матрица со столбцами пользователь-элемент-реакция, которые нужно удалить
        :return: pd.DataFrame, матрица интеракций после удаления рейтингов
        """
        res = interactions.copy()
        renamed_df = df.rename(columns={self.user_col: 'user_id', self.item_col: 'item_id'})
        for index, row in renamed_df.iterrows():
            user = row.user_id
            item = row.item_id
            res[item][user] = 0
        return res

    def _train_valid_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
          Функция для разделенения на train и test DataFrame.
          В тестовой выборке содержится по 1 последнему оцененному элементу для каждого пользователя
        """
        res_df = df.copy()
        res_df[self.sorted_actions_col] = df.groupby([self.user_col])[self.sorting_col].rank(method='first',
                                                                                             ascending=False)
        train_df = res_df[res_df[self.sorted_actions_col] > 4].drop(columns=[self.sorted_actions_col, self.sorting_col])
        valid_df = res_df[res_df[self.sorted_actions_col].isin([3, 4])].drop(
            columns=[self.sorted_actions_col, self.sorting_col])
        test_df = res_df[res_df[self.sorted_actions_col].isin([1, 2])].drop(
            columns=[self.sorted_actions_col, self.sorting_col])
        return train_df, valid_df, test_df

    def _train_valid_test_split_per(self, df: pd.DataFrame, per: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
          Разделение датасета на train, valid, test части.
          Деление выполняется по каждому пользователю. В тестовый датасет попадают последние(хронологически) записи о нем.
          :param df: pd.DataFrame, датасет
          :param per: float, число в из интеравала (0, 1), размер тестовой и валидационной выборок в процентах
          :return: тренировочный, валидационный и тестовый датасеты.
          :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        """
        assert per >= 0 or per <= 1
        groups = dict(tuple(df.groupby(by=self.user_col)))
        train = []
        valid = []
        test = []
        for num, group in groups.items():
            sorted_group = group.sort_values(by=self.sorting_col, ascending=False)
            interactions_count = sorted_group.shape[0]
            percent_count = int(interactions_count * per)
            train_count = interactions_count - 2 * percent_count
            valid_test = sorted_group.head(2 * percent_count)
            train_gr = sorted_group.tail(train_count)
            valid_gr = valid_test.tail(percent_count)
            test_gr = valid_test.head(percent_count)
            train.append(train_gr)
            valid.append(valid_gr)
            test.append(test_gr)
        train_df = pd.concat(train, ignore_index=True).drop(columns=[self.sorting_col])
        valid_df = pd.concat(valid, ignore_index=True).drop(columns=[self.sorting_col])
        test_df = pd.concat(test, ignore_index=True).drop(columns=[self.sorting_col])
        return train_df, valid_df, test_df

    def _train_test_split_per(self, df: pd.DataFrame, per: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
          Разделение датасета на train и test части.
          Деление выполняется по каждому пользователю. В тестовый датасет попадают последние(хронологически) записи о нем.
          :param df: pd.DataFrame, датасет
          :param per: float, число в из интеравала (0, 1), размер тестовой выборки в процентах
          :return: тренировочный и тестовый датасеты.
          :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        assert per >= 0 or per <= 1
        groups = dict(tuple(df.groupby(by=self.user_col)))
        train = []
        test = []
        for num, group in groups.items():
            sorted_group = group.sort_values(by=self.sorting_col, ascending=False)
            interactions_count = sorted_group.shape[0]
            percent_count = int(interactions_count * per)
            train_count = interactions_count - percent_count
            train_gr = sorted_group.tail(train_count)
            test_gr = sorted_group.head(percent_count)
            train.append(train_gr)
            test.append(test_gr)
        train_df = pd.concat(train, ignore_index=True).drop(columns=[self.sorting_col])
        test_df = pd.concat(test, ignore_index=True).drop(columns=[self.sorting_col])
        return train_df, test_df

    def get_train_valid_test_dataloader(self, df: pd.DataFrame, batch_size: int, per: float = 0.1) -> Tuple[DataLoader, DataLoader]:
        """
          Функция для получения на train и test DataLoader.
          Деление выполняется по каждому пользователю. В тестовый датасет попадают последние(хронологически) записи о нем.
          :param df: pd.DataFrame, датасет с данными
          :param batch_size: int, размер батча
          :param per: float, размер тестовой и валидационной выборок в процентах
          :return: DataLoader-ы с тренировочными, валидационными и тестовыми данными
          :rtype: Tuple[DataLoader, DataLoader]
        """
        if self.normilized_col in df.columns:
            ratings = self.normilized_col
        else:
            ratings = self.rating_col

        interactions = self.interaction_matrix(df)
        tr, v, t = self._train_valid_test_split_per(df, per)
        # make train and valid interactions
        valid_interactions = self._remove_ratings_from_interaction_matrix(interactions, t)
        train_interactions = self._remove_ratings_from_interaction_matrix(valid_interactions, v)

        tr_user = tr[self.user_col].tolist()
        tr_item = tr[self.item_col].tolist()
        tr_rating = tr[ratings].tolist()
        tr_dataset = UserItemRatingDataset(tr_user, tr_item, tr_rating, train_interactions)
        train_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

        v_user = v[self.user_col].tolist()
        v_item = v[self.item_col].tolist()
        v_rating = v[ratings].tolist()
        v_dataset = UserItemRatingDataset(v_user, v_item, v_rating, valid_interactions)
        valid_dataloader = DataLoader(v_dataset, batch_size=batch_size, shuffle=True)

        t_user = t[self.user_col].tolist()
        t_item = t[self.item_col].tolist()
        t_rating = t[ratings].tolist()
        t_dataset = UserItemRatingDataset(t_user, t_item, t_rating, interactions)
        test_dataloader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, valid_dataloader, test_dataloader

    def get_train_test_dataloader(self, df: pd.DataFrame, batch_size: int, per: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
          Функция для получения на train и test DataLoader.
          Деление выполняется по каждому пользователю. В тестовый датасет попадают последние(хронологически) записи о нем.
          :param df: pd.DataFrame, датасет с данными
          :param batch_size: int, размер батча
          :param per: float, размер тестовой и валидационной выборок в процентах
          :return: DataLoader-ы с тренировочными и тестовыми данными
          :rtype: Tuple[DataLoader, DataLoader]
        """
        if self.normilized_col in df.columns:
            ratings = self.normilized_col
        else:
            ratings = self.rating_col

        interactions = self.interaction_matrix(df)
        tr, t = self.train_test_split_per(df, per)
        train_interactions = self._remove_ratings_from_interaction_matrix(interactions, t)

        tr_user = tr[self.user_col].tolist()
        tr_item = tr[self.item_col].tolist()
        tr_rating = tr[ratings].tolist()
        tr_dataset = UserItemRatingDataset(tr_user, tr_item, tr_rating, train_interactions)
        train_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

        t_user = t[self.user_col].tolist()
        t_item = t[self.item_col].tolist()
        t_rating = t[ratings].tolist()
        t_dataset = UserItemRatingDataset(t_user, t_item, t_rating, interactions)
        test_dataloader = DataLoader(t_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
