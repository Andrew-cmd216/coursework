import librosa
import opensmile
import pandas as pd
import numpy as np
import parselmouth

from pathlib import Path
from itertools import chain
from Constants.Paths import PATH_TO_DATASET, PATH_TO_CSV, OUTPUT_PATH, PATH_TO_TEST_DATASET

PATH_TO_TESTING = PATH_TO_DATASET / "wav files 1" / "PD-026-pers-1-present.wav"


class Analyser:

    """
    Class responsible for gathering different types of acoustic data from a single audio file.
    """

    def __init__(self, file: Path, path_to_csv: Path):

        """
        Analyser initialisation

        file: Path - path to the '.wav' file
        path_to_csv: Path - path to .csv containing raw text data
        path_to_marked_csv: Path - path to .csv containing text data with time stamps

        returns None
        """

        self.csv = pd.read_csv(path_to_csv)

        self.file = file

        self.array, self.sr = librosa.load(file)
        self.praat_soundfile = parselmouth.Sound(str(file))\

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        self.open_smile_ver = self.smile.process_file(str(self.file))
        self.pitch = self.praat_soundfile.to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=500
        )

        self.csv = self.csv.set_index('Файл')

        self.matrix_result = []

    def _get_statistic_values(self, values):

        """
        Function responsible for returning statistic data

        values: Any - any type of values

        returns:
        E / Q - Mean to standard deviation ratio
        E - Mean value
        Q - Standard deviation
        """

        e = float(np.mean(values))
        q = float(np.std(values))

        result = (e / q, e, q)
        return result

    """*****************************************************************************************************"""

    def _get_speech_ratio(self):

        """
        Function responsible for returning pause to speech ratio

        returns:
        result: dictionary - dictionary containing the name of statistic data (key) and pause to word ratio (value)
        """

        voice_mean = self.open_smile_ver['MeanVoicedSegmentLengthSec'].to_list()[0]

        voice_dev = self.open_smile_ver['StddevVoicedSegmentLengthSec'].to_list()[0]

        silence_mean = self.open_smile_ver['MeanUnvoicedSegmentLength'].to_list()[0]

        silence_dev = self.open_smile_ver['StddevUnvoicedSegmentLength'].to_list()[0]

        result = {
            'Voiced segments mean to dev': voice_mean / voice_dev,
            'Voiced segments mean': voice_mean,
            'Voiced segments dev': voice_dev,
            'Unvoiced segments mean to dev': silence_mean / silence_dev,
            'Unvoiced segments mean': silence_mean,
            'Unvoiced segments dev': silence_mean
        }
        return result

    def _get_speech_tempo(self):

        """

        """

        name = str(self.file.name)
        text = str(self.csv.loc[name])
        text_split_by_words = text.split()

        time = self.array.shape[0] / self.sr

        result = {
           'Speech tempo (words per minute)': float(len(text_split_by_words) / time)
        }

        return result

    def get_prosodic_values(self):

        """

        """

        tempo = self._get_speech_tempo()
        pause_to_speech_ratio = self._get_speech_ratio()

        result = (tempo, pause_to_speech_ratio)

        return result

    """*****************************************************************************************************"""

    def _get_f0(self):

        """

        """

        pitch_time = self._get_statistic_values(self.pitch.selected_array['frequency'])

        result = {
            "F0 mean to dev": pitch_time[0],
            "F0 mean": pitch_time[1],
            "F0 dev": pitch_time[2],
        }

        return result

    def _get_voice_range(self):

        """

        """

        pitch_time = self.pitch.selected_array['frequency']

        voice_range = float(max(pitch_time) - min(pitch_time))

        return {
            'Voice_range (hz)': voice_range
        }

    def _get_tremor_deviation(self):

        """

        """

        tremor = self.open_smile_ver[ 'jitterLocal_sma3nz_stddevNorm'].to_list()[0]

        result = {
            'Voice tremor deviation': tremor
        }

        return result

    def _get_harmonic_ratio(self):

        """

        """

        h1_h2_mean = self.open_smile_ver['logRelF0-H1-H2_sma3nz_amean'].to_list()[0]
        h1_h2_dev = self.open_smile_ver['logRelF0-H1-H2_sma3nz_stddevNorm'].to_list()[0]

        result = {
            'H1/H2 mean to dev': h1_h2_mean / h1_h2_dev,
            'H1/H2 mean': h1_h2_mean,
            'H1/H2 dev': h1_h2_dev
        }

        return result

    def _get_H1_A3_ratio(self):

        """

        """

        h1_a3_mean = self.open_smile_ver['logRelF0-H1-A3_sma3nz_amean'].to_list()[0]
        h1_a3_dev = self.open_smile_ver['logRelF0-H1-A3_sma3nz_stddevNorm'].to_list()[0]

        result = {
            'H1/A3 mean to dev': h1_a3_mean / h1_a3_dev,
            'H1/A3 mean': h1_a3_mean,
            'H1/A3 dev': h1_a3_dev
        }

        return result

    def get_intonation_values(self):

        """

        """

        f0 = self._get_f0()
        voice_range = self._get_voice_range()
        tremor = self._get_tremor_deviation()
        h1_h2 = self._get_harmonic_ratio()
        h1_a3 = self._get_H1_A3_ratio()

        result = (f0, voice_range, tremor, h1_h2, h1_a3)

        return result

    """*****************************************************************************************************"""

    def _get_MFCC(self):

        """
        Function responsible for returning MFCC 1-13

        returns:
        result: dictionary - dictionary containing mean, dev and mean to dev MFCC data
        """

        mfcc1_mean = self.open_smile_ver['mfcc1V_sma3nz_amean'].to_list()[0]
        mfcc1_dev = self.open_smile_ver['mfcc1V_sma3nz_stddevNorm'].to_list()[0]

        mfcc2_mean = self.open_smile_ver['mfcc2V_sma3nz_amean'].to_list()[0]
        mfcc2_dev = self.open_smile_ver['mfcc2V_sma3nz_stddevNorm'].to_list()[0]

        mfcc3_mean = self.open_smile_ver['mfcc3V_sma3nz_amean'].to_list()[0]
        mfcc3_dev = self.open_smile_ver['mfcc3V_sma3nz_stddevNorm'].to_list()[0]

        mfcc4_mean = self.open_smile_ver['mfcc4V_sma3nz_amean'].to_list()[0]
        mfcc4_dev = self.open_smile_ver['mfcc4V_sma3nz_stddevNorm'].to_list()[0]

        result = {
            "MFCC 1 mean to dev": mfcc1_mean / mfcc1_dev,
            "MFCC 1 mean": mfcc1_mean,
            "MFCC 1 dev": mfcc1_dev,

            "MFCC 2 mean to dev": mfcc2_mean / mfcc2_dev,
            "MFCC 2 mean": mfcc2_mean,
            "MFCC 2 dev": mfcc2_dev,

            "MFCC 3 mean to dev": mfcc3_mean / mfcc3_dev,
            "MFCC 3 mean": mfcc3_mean,
            "MFCC 3 dev": mfcc3_dev,

            "MFCC 4 mean to dev": mfcc4_mean / mfcc4_dev,
            "MFCC 4 mean": mfcc4_mean,
            "MFCC 4 dev": mfcc4_dev,
            }

        return result

    def _get_formant(self):

        """
        Function responsible for returning formant

        returns:
        result: dictionary - dictionary containing mean, dev and mean to dev formant data
        """

        formant = self.praat_soundfile.to_formant_burg(time_step=0.1)

        times = np.arange(0, self.praat_soundfile.duration, 0.01)

        formant_1 = []
        formant_2 = []
        formant_3 = []

        for t in times:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            formant_1.append(f1)
            formant_2.append(f2)
            formant_3.append(f3)

        stats_f1 = self._get_statistic_values(formant_1)

        stats_f2 = self._get_statistic_values(formant_2)

        stats_f3 = self._get_statistic_values(formant_3)

        result = {
            "F1 mean to dev": stats_f1[0],
            "F1 mean": stats_f1[1],
            "F1 dev": stats_f1[2],

            "F2 mean to dev": stats_f2[0],
            "F2 mean": stats_f2[1],
            "F2 dev": stats_f2[2],

            "F3 mean to dev": stats_f3[0],
            "F3 mean": stats_f3[1],
            "F3 dev": stats_f3[2]
        }
        return result

    def get_acoustic_values(self):

        """
        Function responsible for returning acoustic data

        returns:
        mfcc: dictionary - MFCC dictionary
        formant: dictionary - formant dictionary
        """

        mfcc = self._get_MFCC()
        formant = self._get_formant()

        result = (mfcc, formant)

        return result

    """*****************************************************************************************************"""
    def _get_amplitude_spikes(self):

        """

        """

        peaks = self.open_smile_ver["loudnessPeaksPerSec"].to_list()[0]

        return {
            'Loudness peaks per second': peaks
        }

    def _get_Hammarberg(self):

        """

        """

        ham_mean = self.open_smile_ver["hammarbergIndexV_sma3nz_amean"].to_list()[0]

        ham_dev = self.open_smile_ver["hammarbergIndexV_sma3nz_stddevNorm"].to_list()[0]

        mean_to_dev = ham_mean / ham_dev

        result = {
            'Hammarberg mean to dev': mean_to_dev,
            'Hammarberg mean': ham_mean,
            'Hammarberg dev': ham_dev
        }

        return result

    def get_amplitude_values(self):

        """

        """
        spikes = self._get_amplitude_spikes()
        hammarberg = self._get_Hammarberg()

        result = (spikes, hammarberg)

        return result

    """*****************************************************************************************************"""

    def get_HNR_values(self):

        """

        """

        hnr_mean = self.open_smile_ver["HNRdBACF_sma3nz_amean"].to_list()[0]

        hnr_dev = self.open_smile_ver["HNRdBACF_sma3nz_stddevNorm"].to_list()[0]

        mean_to_dev = hnr_mean / hnr_dev

        result = [{
            'HNR mean to dev': mean_to_dev,
            'HNR mean': hnr_mean,
            'HNR dev': hnr_dev
        }]

        return result

    """*****************************************************************************************************"""

    def analyse_doc(self):

        """

        """

        result = pd.DataFrame()

        prosody = self.get_prosodic_values()
        acoustic = self.get_acoustic_values()
        amplitude = self.get_amplitude_values()
        intonation = self.get_intonation_values()
        hnr = self.get_HNR_values()

        dicts = [prosody, acoustic, amplitude, intonation, hnr]

        dicts_unpacked = []

        for val in dicts:
            dicts_unpacked.extend(val)

        chars_combined = dict(chain.from_iterable(d.items() for d in dicts_unpacked))

        row = pd.DataFrame(chars_combined, index=[str(self.file.name)])

        result = pd.concat([result, row])

        return result

class MatrixBuilder:

    """

    """

    def __init__(self, path_to_dataset: Path, path_to_csv: Path):
        self.__path_to_dataset = path_to_dataset
        self.__path_to_csv = path_to_csv
        self.matrix_result = []


    def build_matrix(self):

        """

        """

        matrix = pd.DataFrame()

        for n_file in self.__path_to_dataset.iterdir():

            analyser = Analyser(n_file, self.__path_to_csv)
            data = analyser.analyse_doc()

            matrix = pd.concat([matrix, data])

        matrix.index.name = "Файл"

        output_path = OUTPUT_PATH / f'{str(self.__path_to_dataset.name)}.csv'

        matrix.to_csv(output_path)

        print("Done!")


matrix_builder = MatrixBuilder(path_to_dataset=PATH_TO_TEST_DATASET, path_to_csv=PATH_TO_CSV)
matrix_builder.build_matrix()
