# from src.preprocessing import PitchShift, MelSpectrogram
import os
import sys
# import librosa
sys.path.append(os.getcwd() + "/src/")
# from dataset import HierarchicalDataset, LabeledDataset


def main():
    # hier = HierarchicalDataset(root="data/hierarchical", data_type="image")
    # print(hier[0])
    # file, sr = librosa.load("negawatt.wav")
    # shifted = PitchShift(pitch_factor=2.0, sample_rate=sr)
    # shifted_file = shifted(file)
    # sf.write("negashift.wav", shifted_file, sr, format="wav")
    # mel = MelSpectrogram(sample_rate=sr, file_name="mel.png")
    # mel(file)

    pass


if __name__ == "__main__":
    main()
