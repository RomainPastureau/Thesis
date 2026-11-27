
% Create speech in noise wav files with 10 s no-noise, 10 s ramp and 300 s of speech in noise

norm_fact = 0.08;  % noralization factor to keep same power across recordings and (typically) avoid clipping
FS = 48000;

path_noise = 'D:\Noise\';
path_audios = 'D:\Audio\';
norm_path = strcat(path_audios, '05_Noised_audio\No_noise_normalized\');
if ~exist(norm_path, 'dir')
    mkdir(norm_path)
end
subjects = ["K01\", "K02\", "K03\", "K04\", "K05\", "K06\"];
SNRs = [-15 -10 -5 0 1 2 3 4 5 10 15];
folders = ("");
for i = 1:size(SNRs,2)
    folders(i) = strcat("Noised_audio_",int2str(SNRs(i)),"\");
end

% create 310 s of background noise
noise = zeros(1, FS*300);

listing = dir(strcat(path_noise, 'Cocktail party 2 - Cut, 48000\'));
for n_text = 3:8
    wavfile = strcat(path_noise, 'Cocktail party 2 - Cut, 48000\', listing(n_text).name) % path to each of the wav files of a single talker used as element of the background noise
    [Y, FS] = audioread(wavfile);
    Y = Y(1:300*FS); % keep only 180 s
    Y = Y/sqrt(mean(Y.^2))*norm_fact;
    noise = noise + Y;
end

noise = noise/sqrt(mean(noise.^2))*norm_fact;
noise = noise(120*FS:end);
%noise(1:10*FS) = noise(1:10*FS).*(1:10*FS)'/10/FS; % slowly increase the noise from 0 to 10 s
wavfile = strcat(path_noise, 'Cocktail party 2 - Merged\Cocktail_party_2.wav'); % output noise file
audiowrite(wavfile, noise, FS);

% no noise
for i = 1:6
    listing = dir(strcat(path_audios, '04_Trimmed_audio\', convertStringsToChars(subjects(i))));
    if ~exist(strcat(norm_path, convertStringsToChars(subjects(i))), 'dir')
        mkdir(strcat(norm_path, convertStringsToChars(subjects(i))))
    end
    for n_text = 3:(size(listing,1))
        wavfile = strcat(path_audios, '04_Trimmed_audio\', convertStringsToChars(subjects(i)), listing(n_text).name); % wav file of a recording to present without noise
        [Y,FS] = audioread(wavfile);
        limit = min(size(Y, 1), 300*FS);
        Y = Y(1:limit);
        Y = Y/sqrt(mean(Y.^2))*norm_fact;

        wavfile = strcat(norm_path, convertStringsToChars(subjects(i)), listing(n_text).name) % name of the ouput file (normalized amplitude)
        audiowrite(wavfile,Y,FS);
    end
end

% noise conditions
SNR_fact = 10.^(SNRs/20); % number in between [] are SNRs in dB.
if size(noise, 1) == 1
    noise = transpose(noise);
end

for n = 1:size(SNRs,2)
    for i = 1:6
        listing = dir(strcat(norm_path, convertStringsToChars(subjects(i))));
        if ~exist(strcat(path_audios, "05_Noised_audio/", folders(n), convertStringsToChars(subjects(i))), 'dir')
            mkdir(strcat(path_audios, "05_Noised_audio/", folders(n), convertStringsToChars(subjects(i))));
        end
        for n_text = 3:(size(listing,1))
            wavfile = strcat(norm_path, convertStringsToChars(subjects(i)), listing(n_text).name); % wav file of a recording to present with noise
            [Y,FS] = audioread(wavfile);
            limit = min(size(Y, 1), 300*FS);
            Y = Y(1:limit);
            Y = Y/sqrt(mean(Y.^2))*norm_fact*SNR_fact(n);
            Y = Y + noise(1:limit);
            Y = Y/sqrt(mean(Y(20*FS+1:end).^2))*norm_fact; % use only if you want to keep sound + noise power constant across conditions
            wavfile = strcat(path_audios, '05_Noised_audio\', convertStringsToChars(folders(n)), convertStringsToChars(subjects(i)), listing(n_text).name) % ouput wav file
            audiowrite(wavfile,Y(1:limit),FS);
        end
    end
end
