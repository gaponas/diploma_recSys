from pandas import read_csv

# TODO: удалить реальный путь на название+папку
MOVIELENS_100K_RATINGS = "/home/aleksandra/MainInfo/magistr/diploma/diploma_recSys/data/datasets/u.data"
""" Путь к датасету MovieLens100k """
MOVIELENS_1M_RATINGS = "/home/aleksandra/MainInfo/magistr/diploma/diploma_recSys/data/datasets/ratings.dat"
""" Путь к датасету MovieLens1m """


class UploadData:
    """
    Класс для загрузки датасетов
    """

    @staticmethod
    def movielens_1m_df():
        return read_csv(MOVIELENS_1M_RATINGS, sep="::", names=["UserId", "MovieId", "Rating", "Timestamp"])

    @staticmethod
    def movielens_100k_df():
        return read_csv(MOVIELENS_100K_RATINGS, sep="\t", names=["UserId", "MovieId", "Rating", "Timestamp"])
