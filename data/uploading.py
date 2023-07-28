from pandas import read_csv

MOVIELENS_100K_RATINGS = "/content/drive/MyDrive/diploma/ml-100k/u.data"
""" Путь к датасету MovieLens100k """
MOVIELENS_1M_RATINGS = "/content/drive/MyDrive/diploma/ml-1m/ratings.dat"
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
