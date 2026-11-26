from krajjat import Audio

from helper_functions import *
from constants import *

def generate_audio_derivative(kind, subject, recording, conditions, verbosity=1, **kwargs):
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    path_derivative = op.join(PATH_AUDIO_DERIVATIVE, kind, subject, recording + ".pkl")
    os.makedirs(op.dirname(path_derivative), exist_ok=True)
    condition = conditions[subject][recording]

    print(f"{kind} · {subject} · {recording}")
    audio = Audio(path_audio, name=recording, condition=condition, verbosity=verbosity)
    derivative = audio.get_derivative(kind, verbosity=verbosity, **kwargs).resample(FS, verbosity=verbosity)
    print(f"\t\tSamples derivative: {derivative.get_number_of_samples()}, duration {derivative.get_duration():.2f} s.".replace(",", " "))
    derivative.save_figure(path_derivative)

def plot_derivative(kind, subject, recording, verbosity=1, **kwargs):
    path_audio = op.join(PATH_AUDIO_PREPROCESSED, subject, recording + ".wav")
    audio = Audio(path_audio, name=recording, verbosity=verbosity)
    derivative = audio.get_derivative(kind, verbosity=verbosity, **kwargs).resample(FS, verbosity=verbosity)

    import matplotlib.pyplot as plt
    plt.plot(derivative.get_timestamps(), derivative.get_samples())
    plt.show()

if __name__ == "__main__":
    conditions = get_conditions()

    # # All recordings
    for derivative in DERIVATIVES:
        for subject in ALL_SUBJECTS:
            path_audio_subject = op.join(PATH_AUDIO_PREPROCESSED, subject)
            recordings = os.listdir(path_audio_subject)
            for recording in recordings:
                generate_audio_derivative(derivative, subject, op.splitext(recording)[0], conditions, verbosity=0,
                                          filter_over=CUTOFF_FREQ_HIGH)

    # Single recording
    # generate_audio_derivative("envelope", "K01", "R004", conditions, filter_over=4)

    # Test
    # plot_derivative("f1", "K01", "R004", verbosity=1, filter_over=4)
