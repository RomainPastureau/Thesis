from krajjat import Audio

from helper_functions import *
from constants import *

def generate_audio_derivative(kind, subject, recording, verbosity=1, **kwargs):
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_derivative = op.join(PATH_AUDIO_DERIVATIVE, kind, subject, recording + ".pkl")
    os.makedirs(op.dirname(path_derivative), exist_ok=True)

    print(f"{kind} · {subject} · {recording}")
    audio = Audio(path_audio, name=recording, verbosity=verbosity)
    derivative = audio.get_derivative(kind, verbosity=verbosity, **kwargs).resample(FS, verbosity=verbosity)
    print(f"\t\tSamples derivative: {derivative.get_number_of_samples()}, duration {derivative.get_duration():.2f} s.".replace(",", " "))
    derivative.save(path_derivative)

def plot_derivative(kind, subject, recording, verbosity=1, **kwargs):
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    audio = Audio(path_audio, name=recording, verbosity=verbosity)
    derivative = audio.get_derivative(kind, verbosity=verbosity, **kwargs).resample(FS, verbosity=verbosity)

    import matplotlib.pyplot as plt
    plt.plot(derivative.get_timestamps(), derivative.get_samples())
    plt.show()

if __name__ == "__main__":

    # # All recordings
    for derivative in DERIVATIVES:
        for subject in ALL_SUBJECTS:
            path_audio_subject = op.join(PATH_AUDIO_PREPROCESSED, subject)
            recordings = os.listdir(path_audio_subject)
            for recording in recordings:
                generate_audio_derivative(derivative, subject, op.splitext(recording)[0], verbosity=0, filter_above=CUTOFF_FREQ_HIGH)

    # Single recording
    generate_audio_derivative("envelope", "S001", "R001", filter_above=CUTOFF_FREQ_HIGH)

    # Test
    plot_derivative("envelope", "S001", "R001", verbosity=1, filter_above=CUTOFF_FREQ_HIGH)
