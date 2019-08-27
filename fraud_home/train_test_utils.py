import pandas as pd
from sklearn.model_selection import train_test_split


class train_test:

    @staticmethod
    def training_test_valid_unbalanced(self, anormal_file):
        """
        Separate between training, test and valid using the next proportions:
        Training 70%
        Test 15%
        Valid 15%
        Also it keeps the same proportion between Fraud class inside Test an Valid.
        However, it excludes every fraud claim in the Train Set.
        """
        normal = pd.read_csv(self, delimiter=';')
        anomaly = pd.read_csv(anormal_file, delimiter=';')
        # normal = shuffle(normal)
        normal = normal.reset_index(drop=True)
        # anomaly = shuffle(anomaly)
        anomaly = anomaly.reset_index(drop=True)
        train, normal_test, _, _ = train_test_split(normal, normal, test_size=.3, random_state=42)

        normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test, test_size=.5, random_state=42)
        anormal_valid, anormal_test, _, _ = train_test_split(anomaly, anomaly, test_size=.5, random_state=42)

        train = train.reset_index(drop=True)
        valid = normal_valid.append(anormal_valid).sample(frac=1).reset_index(drop=True)
        test = normal_test.append(anormal_test).sample(frac=1).reset_index(drop=True)

        return train, valid, test

    @staticmethod
    def training_test_unbalanced(self, anormal_file):
        """
        Separate between training and Test using the next proportions:
        Training 70%
        Test 30%
        Also it keeps the same proportion between Fraud class.
        However, it excludes every fraud claim in the Train Set.
        """
        normal = pd.read_csv(self, delimiter=';')
        anormal = pd.read_csv(anormal_file, delimiter=';')
        # normal = shuffle(normal)
        normal = normal.reset_index(drop=True)
        # anomaly = shuffle(anormal)
        anormal = anormal.reset_index(drop=True)

        train_normal, test_normal, _, _ = train_test_split(normal, normal, test_size=.3, random_state=42)
        train_anormal, test_anormal, _, _ = train_test_split(anormal, anormal, test_size=.3, random_state=42)

        train = train_normal.append(train_anormal).sample(frac=1).reset_index(drop=True)
        test = test_normal.append(test_anormal).sample(frac=1).reset_index(drop=True)

        return train, test

    @staticmethod
    def inverse_training_test_valid(self, anormal_file):
        """
        Separate between training, test and valid using the next proportions:
        Training 70%
        Test 15%
        Valid 15%
        The difference is that it only includes anomaly cases inside the Train Set.
        """
        normal = pd.read_csv(self, delimiter=';')
        anomaly = pd.read_csv(anormal_file, delimiter=';')
        # normal = shuffle(normal)
        normal = normal.reset_index(drop=True)
        # anomaly = shuffle(anomaly)
        anomaly = anomaly.reset_index(drop=True)

        train, anormal_test, _, _ = train_test_split(anomaly, anomaly, test_size=.5, random_state=42)

        anormal_valid, anormal_test, _, _ = train_test_split(anormal_test, anormal_test, test_size=.5, random_state=42)
        normal_valid, normal_test, _, _ = train_test_split(normal, normal, test_size=.5, random_state=42)

        train = train.reset_index(drop=True)
        valid = anormal_valid.append(normal_valid).sample(frac=1).reset_index(drop=True)
        test = anormal_test.append(normal_test).sample(frac=1).reset_index(drop=True)

        return train, valid, test

    @staticmethod
    def training_test_valid(self, anomaly):
        """
        Separate between training, test and valid using the next proportions:
        Training 70%
        Test 15%
        Valid 15%
        Here, we include in the Training Set either normal cases and anormal cases using the proportions
        derivated from the original distribution.
        Then we split between Test and Valid using the same original proportions.
        """

        # normal = shuffle(normal)
        normal = self.reset_index(drop=True)
        # anomaly = shuffle(anomaly)
        anomaly = anomaly.reset_index(drop=True)

        normal_train, normal_test, _, _ = train_test_split(normal, normal, test_size=.3, random_state=42)
        anormal_train, anormal_test, _, _ = train_test_split(anomaly, anomaly, test_size=.3, random_state=42)
        normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test, test_size=.5, random_state=42)
        anormal_valid, anormal_test, _, _ = train_test_split(anormal_test, anormal_test, test_size=.5, random_state=42)

        train = normal_train.append(anormal_train).sample(frac=1).reset_index(drop=True)
        valid = normal_valid.append(anormal_valid).sample(frac=1).reset_index(drop=True)
        test = normal_test.append(anormal_test).sample(frac=1).reset_index(drop=True)

        return train, valid, test

    @staticmethod
    def training_test(self, anomaly):
        """
        Separate between training, test and valid using the next proportions:
        Training 70%
        Test 15%
        Valid 15%
        Here, we include in the Training Set either normal cases and anormal cases using the proportions
        derivated from the original distribution.
        Then we split between Test and Valid using the same original proportions.
        """
        # normal = shuffle(normal)
        normal = self.reset_index(drop=True)
        # anomaly = shuffle(anomaly)
        anomaly = anomaly.reset_index(drop=True)

        normal_train, normal_test, _, _ = train_test_split(normal, normal, test_size=.3, random_state=42)
        anormal_train, anormal_test, _, _ = train_test_split(anomaly, anomaly, test_size=.3, random_state=42)

        train = normal_train.append(anormal_train).sample(frac=1).reset_index(drop=True)
        test = normal_test.append(anormal_test).sample(frac=1).reset_index(drop=True)

        return train, test

