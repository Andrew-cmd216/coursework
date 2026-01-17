import pandas as pd
import torchaudio
from pathlib import Path
import string

ROOT_PATH = Path(__file__).parent
PATH_TO_DATASET = ROOT_PATH / "Аудиофайлы" / "Аудиофайлы" / "NLP+PD"

class Verifyer:

    """

    """

    def __init__(self, path_to_dataset: Path):
        '''

        '''
        self.__path = path_to_dataset
        self._length = 0
        self._avg_length = 0
        self._trouble_files = {
            "sample_rate" : [],
            "channels": []
        }
        self.__chars = {
            'sample_rate': 48000,
            'num_channels': 1
        }
        self.__directories = []
        self.__fill_directories()

    def __fill_directories(self):
        for rep in self.__path.iterdir():
           if "wav files" in rep.name:
               self.__directories.append(rep)

    def verification(self):
        sum_length = 0
        len_dir = 0
        for directory in self.__directories:
            direc = list(directory.iterdir())
            len_dir += len(direc)
            for file in direc:
                array, sr = torchaudio.load(file)
                num_channels = array.shape[0]
                if sr != self.__chars['sample_rate']:
                    self._trouble_files["sample_rate"].append((file, sr))
                if num_channels != self.__chars['num_channels']:
                    self._trouble_files["channels"].append((file, num_channels))
                sum_length += ((array.shape[1]) / sr)
        self._length = sum_length / 60
        self._avg_length = sum_length / len_dir / 60

        print(f'{self._avg_length}\n{self._length}\n{self._trouble_files}'
              f'\n{len(self._trouble_files["sample_rate"])}\n{len(self._trouble_files["channels"])}'
              f'\n{min(self._trouble_files["sample_rate"], key=lambda x: x[1])}')

class DataOrganiser:

    """

    """

    def __init__(self, path_to_csv):
        self.csv = pd.read_csv(path_to_csv)
        self._new_csv = pd.DataFrame

    def organise_csv(self, path_to_save):
        new_csv = pd.DataFrame()
        for element in self.csv.itertuples(index=False):

            normalised_path = str(element[0]).split('/')[1]
            normalised_path = normalised_path.replace('txt', 'wav')

            text = element[1]
            text = text.split('\n')
            text_new = [x.split(']:')[-1] for x in text]
            text_new = ' '.join(text_new)
            translator = str.maketrans('', '', string.punctuation)
            text_cleaned = text_new.translate(translator).lower()
            row = pd.DataFrame({'Текст': text_cleaned},
                               index=[normalised_path])
            new_csv = pd.concat([new_csv, row])
        new_csv.to_csv(path_to_save)


organiser = DataOrganiser(f'{PATH_TO_DATASET}\\all_transcriptions.csv')
organiser.organise_csv(f'{PATH_TO_DATASET}\\transcriptions_clean.csv')

#verificator = Verifyer(PATH_TO_DATASET)
#verificator.verification()
