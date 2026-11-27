%% BODYLINGUAL COMP (ROMAIN + DAVID)
% Cries in Matlab %

%% Preparation and path selection
clear; clc;
verbose = true; % Put to false if you don't want to get any feedback in the console.

dir_exp = 'D:/BLComp/';
cd(dir_exp);
addpath('C:/Psychtoolbox/');
path_stim = strcat(dir_exp, 'Stimuli/');
path_results = strcat(dir_exp, 'Results/');
path_questions = strcat(dir_exp, 'Questions.txt');
path_instructions = strcat(dir_exp, 'Instructions.txt');
path_images_instructions = strcat(dir_exp, 'Images_Instructions/');

% Put to true to:
% - Remove the trigger sending
% - Control with the keyboard instead of the buttons
debug = true;

%% Buttons definition
red = 65521;
yellow = 65522; 
green = 65524;
blue = 65528;

%% Init DataPixx
InitializePsychSound(1);
% Init parallel port
if debug == false
    ioObj = io64;
    status = io64(ioObj);
    address = hex2dec('CFB8'); % used to be BFB8
    io64(ioObj, address, 0) % blank register
    Init_DataPiXX_n_port();
end

%% Randomization seed generation
rng('shuffle');

%% Question loading and answer randomization
file_questions = fopen(path_questions, 'r', 'n', 'UTF-8');
questions = {81};
for i = 1:81
    q = [];
    q.id = fgetl(file_questions);
    q.question = fgetl(file_questions);
    q.good_answer = fgetl(file_questions);
    q.bad_answer1 = fgetl(file_questions);
    q.bad_answer2 = fgetl(file_questions);
    q.bad_answer3 = fgetl(file_questions);
    q.answers_rand = {q.good_answer, q.bad_answer1, q.bad_answer2, q.bad_answer3};
    q.answers_rand = q.answers_rand(randperm(length(q.answers_rand)));
    linebreak = fgetl(file_questions);
    questions{i} = q;
end
fclose(file_questions);
    
if verbose == true
    fprintf('Questions loaded correctly.\n');
end

%% Instructions loading
last_instruction_screen = 0; % Saves instruction screen
file_instructions = fopen(path_instructions, 'r', 'n', 'UTF-8');

% Get number of lines
number_of_lines = 0;
while true
  if ~ischar( fgetl(file_instructions) ); break; end
  number_of_lines = number_of_lines + 1;
end
fclose(file_instructions);

file_instructions = fopen(path_instructions, 'r', 'n', 'UTF-8');

% Get instructions
inst = {};
screentext = '';
j = 1;
for i = 1:number_of_lines
    line = fgetl(file_instructions);
    if isempty(line)
        inst{j} = screentext;
        j = j + 1;
        screentext = '';
    elseif i == number_of_lines
        screentext = strcat(screentext, '\n\n', line);
        inst{j} = screentext;
        j = j + 1;
        screentext = '';        
    else
        screentext = strcat(screentext, '\n\n', line);
    end
end
fclose(file_instructions);

% Load the images
image_files = dir([path_images_instructions, '\*.png']);
images_opened = cell(5,1);
images_positions = [];
current_image = 0;

for image = 1:size(image_files)
    images_opened{image, 1} = imread(strcat(path_images_instructions, image_files(image).name));
    %split_extension = split(image_files(image).name, '.');
    %split_filename = split(split_extension(1), '_');
    split_chars = num2cell(image_files(image).name);
    images_positions(image) = str2num(char(split_chars(13)));
end

if verbose == true
    fprintf('Instructions loaded correctly.\n');
end

%% Input subject
%subject = str2double(input('Subject: ', 's'));
subject = input('Subject (Participa ID): ', 's');
KbName('UnifyKeyNames')

list_number = -1;
files_in_results = dir([path_results, '\*.mat']);
default_list = num2str(length(files_in_results) + 1);
while (0 > list_number) || (list_number > 30)
    list_number = input(['Force subject list (1-30)? If no, type 0, and the selected list will be ', default_list, ': ']);
end

%% Getting participants' list of videos

% Get participant number depending on number of files in the results folder
if list_number == 0
    list_number = mod(length(files_in_results) + 1, 30);
end

% Get the videos for the participant
[~,~,stimuli_list_all_subjects] = xlsread([dir_exp, 'stimuli_list.xlsx']);
first_column = cell2mat(stimuli_list_all_subjects(:,1));
stimuli_subject = stimuli_list_all_subjects(first_column == list_number, :);
stimuli_subject = stimuli_subject(:,2);

if verbose == true
    fprintf(['Stimuli list number ', num2str(list_number), ' of 30 loaded correctly.\n']);
end

%% Creating conditions sublists

% Conditions
sublist_audio_unmasked = stimuli_subject(cellfun(@(x) strncmpi(x, 'AO_U_XX', length('AO_U_XX')), stimuli_subject),:);
sublist_audio_masked = stimuli_subject(cellfun(@(x) strncmpi(x, 'AO_M_XX', length('AO_M_XX')), stimuli_subject),:);
sublist_video_unmasked = stimuli_subject(cellfun(@(x) strncmpi(x, 'VO_U_BD', length('VO_U_BD')), stimuli_subject),:);
sublist_video_masked = stimuli_subject(cellfun(@(x) strncmpi(x, 'VO_M_BD', length('VO_M_BD')), stimuli_subject),:);
sublist_video_skeleton = stimuli_subject(cellfun(@(x) strncmpi(x, 'VO_X_SK', length('VO_X_SK')), stimuli_subject),:);
sublist_video_scrambled = stimuli_subject(cellfun(@(x) strncmpi(x, 'VO_X_SC', length('VO_X_SC')), stimuli_subject),:);
sublist_audiovisual_unmasked = stimuli_subject(cellfun(@(x) strncmpi(x, 'AV_U_BD', length('AV_U_BD')), stimuli_subject),:);
sublist_audiovisual_masked = stimuli_subject(cellfun(@(x) strncmpi(x, 'AV_M_BD', length('AV_M_BD')), stimuli_subject),:);
sublist_audiovisual_skeleton = stimuli_subject(cellfun(@(x) strncmpi(x, 'AV_X_SK', length('AV_X_SK')), stimuli_subject),:);
sublist_audiovisual_scrambled = stimuli_subject(cellfun(@(x) strncmpi(x, 'AV_X_SC', length('AV_X_SC')), stimuli_subject),:);

% Randomized conditions
rand_sublist_audio_unmasked = sublist_audio_unmasked(randperm(length(sublist_audio_unmasked)));
rand_sublist_audio_masked = sublist_audio_masked(randperm(length(sublist_audio_masked)));
rand_sublist_video_unmasked = sublist_video_unmasked(randperm(length(sublist_video_unmasked)));
rand_sublist_video_masked = sublist_video_masked(randperm(length(sublist_video_masked)));
rand_sublist_video_skeleton = sublist_video_skeleton(randperm(length(sublist_video_skeleton)));
rand_sublist_video_scrambled = sublist_video_scrambled(randperm(length(sublist_video_scrambled)));
rand_sublist_audiovisual_unmasked = sublist_audiovisual_unmasked(randperm(length(sublist_audiovisual_unmasked)));
rand_sublist_audiovisual_masked = sublist_audiovisual_masked(randperm(length(sublist_audiovisual_masked)));
rand_sublist_audiovisual_skeleton = sublist_audiovisual_skeleton(randperm(length(sublist_audiovisual_skeleton)));
rand_sublist_audiovisual_scrambled = sublist_audiovisual_scrambled(randperm(length(sublist_audiovisual_scrambled)));

%% Randomization
rand_stimuli_subject = {80};

for i = 1:8
    order = {'AO_U_XX', 'AO_M_XX', ...
             'VO_U_BD', 'VO_M_BD', 'VO_X_SK', 'VO_X_SC', ...
             'AV_U_BD', 'AV_M_BD', 'AV_X_SK', 'AV_X_SC'};
         
    audio_only_conds = {'AO_U_XX', 'AO_M_XX'};
    video_only_conds = {'VO_U_BD', 'VO_M_BD', 'VO_X_SK', 'VO_X_SC'};
    audiovisual_conds = {'AV_U_BD', 'AV_M_BD', 'AV_X_SK', 'AV_X_SC'};
    
    rand_audio_only_conds = audio_only_conds(randperm(length(audio_only_conds)));
    rand_video_only_conds = video_only_conds(randperm(length(video_only_conds)));
    rand_audiovisual_conds = audiovisual_conds(randperm(length(audiovisual_conds)));
    
    place_audio = randi([1 2]);
    
    if place_audio == 1
        part1 = [rand_audio_only_conds(1), rand_video_only_conds(1), rand_audiovisual_conds(1)];
        part2 = [rand_video_only_conds(2), rand_audiovisual_conds(2)];
    else
        part1 = [rand_video_only_conds(1), rand_audiovisual_conds(1)];
        part2 = [rand_audio_only_conds(1), rand_video_only_conds(2), rand_audiovisual_conds(2)];
    end
    
    place_audio = randi([1 2]);
    
    if place_audio == 1
        part3 = [rand_audio_only_conds(2), rand_video_only_conds(3), rand_audiovisual_conds(3)];
        part4 = [rand_video_only_conds(4), rand_audiovisual_conds(4)];
    else
        part3 = [rand_video_only_conds(3), rand_audiovisual_conds(3)];
        part4 = [rand_audio_only_conds(2), rand_video_only_conds(4), rand_audiovisual_conds(4)];
    end
    
    rand_part1 = part1(randperm(length(part1)));
    rand_part2 = part2(randperm(length(part2)));
    rand_part3 = part3(randperm(length(part3)));
    rand_part4 = part4(randperm(length(part4)));
    
    order_rand = [rand_part1, rand_part2, rand_part3, rand_part4];
    
    for j = 1:10
        switch char(order_rand(j))
            case 'AO_U_XX'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_audio_unmasked(i);
            case 'AO_M_XX'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_audio_masked(i);
            case 'VO_U_BD'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_video_unmasked(i);
            case 'VO_M_BD'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_video_masked(i);
            case 'VO_X_SK'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_video_skeleton(i);
            case 'VO_X_SC'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_video_scrambled(i);
            case 'AV_U_BD'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_audiovisual_unmasked(i);
            case 'AV_M_BD'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_audiovisual_masked(i);
            case 'AV_X_SK'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_audiovisual_skeleton(i);
            case 'AV_X_SC'
                rand_stimuli_subject{(i-1)*10+j} = rand_sublist_audiovisual_scrambled(i);
        end
    end
end

stimuli_subject = rand_stimuli_subject;

if verbose == true
    fprintf('Videos randomized correctly.\n');
end

%% Clean up the variables we don't need
clear -regexp rand_sub* sublist_* part* first_column rand_a* rand_v* stimuli_list_all* split* ord*

%% Screen setup
Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
screenNumber = max(screens);
black = BlackIndex(screenNumber);
white = WhiteIndex(screenNumber);
HideCursor(screenNumber);

% Get screen size
screenResolution = Screen('Resolution', screenNumber);
width = (screenResolution.width);
height = (screenResolution.height);
ratioW = width/1920;
ratioH = height/1080;

[win, windowRect] = PsychImaging('OpenWindow', screenNumber, black, [0 0 width height]);
Screen('BlendFunction', win, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
Screen('Preference', 'TextRenderer', 1);
Screen('Preference', 'TextAntiAliasing', 2);

% Get the size screen and rec
[screenXpixels, screenYpixels] = Screen('WindowSize', win);
[xCenter, yCenter] = RectCenter(windowRect);
screenRect = windowRect;

% Video size
[xCenter, yCenter] = RectCenter(windowRect);
rect = [0 0 width*ratioW height*ratioW];
rectVid = CenterRectOnPointd(rect, xCenter, yCenter);

%% Define variables
nTrials = 81;
data.subject = []; % Subject number
data.number = []; % Trial number (1-81)
data.condition = []; % Condition (XX_X_XX)
data.stimulus = []; % Code of the video (XX_X_XX_000)
data.trigTime = []; % VPixx time at which the video starts
data.question = []; % Question structure
data.resp = []; % Code of the button pressed (100, 200, 300 or 400)
data.button_pressed = []; % Button pressed (red, yellow, green or blue)
data.RT = []; % Time before pressing the button
data.selected_answer = []; % Answer selected for the question

%% Instructions (first batch)
Screen('TextFont', win, 'Arial', 0);
Screen('TextSize', win, 40);

for i = 1:8
    instructions_batch = strcat(inst{i});
    if ismember(i, images_positions)
        current_image = current_image + 1;
        Screen('PutImage', win, images_opened{current_image});  
        instructions_batch = strcat(inst{i}, '\n\n\n\n');
    else
        instructions_batch = strcat(inst{i});
    end
    DrawFormattedText(win, instructions_batch, 'center', 'center', white);
    Screen(win, 'flip');
    if verbose == 1
        fprintf(strcat('Showing instruction screen ', num2str(i), ' of ', num2str(size(inst,2)), '.\n'));
    end
    if debug == false
        meg_KbWait_SN(Inf);
    else
        KbStrokeWait(Inf);
    end
end

%% Experiment

for trial = 1:nTrials
    
    if trial == 2
        for i = 9:11
            instructions_batch = strcat(inst{i});
            if ismember(i, images_positions)
                current_image = current_image + 1;
                Screen('PutImage', win, images_opened{current_image});  
                instructions_batch = strcat(inst{i}, '\n\n\n\n');
            else
                instructions_batch = strcat(inst{i});
            end
            DrawFormattedText(win, instructions_batch, 'center', 'center', white);
            Screen(win, 'flip');
            if verbose == 1
                fprintf(strcat('Showing instruction screen ', num2str(i), ' of ', num2str(size(inst,2)), '.\n'));
            end
            if debug == false
                meg_KbWait_SN(Inf);
            else
                KbStrokeWait(Inf);
            end
        end
    end
    
    % Get variables
    if trial == 1
        stimulus = 'AV_U_BD_081.mp4';
    else
        stimulus = stimuli_subject{trial-1};
    end
    
    condition = stimulus(1:7);
    path_stimulus = strcat(path_stim, condition, '/', stimulus);
    id = stimulus(9:11);
    
    % Save data
    data.subject{trial} = subject;
    data.number{trial} = trial;
    data.condition{trial} = condition;
    data.stimulus{trial} = stimulus;
    
    % Get question
    question = questions(str2num(id));
    question = question{1};
    data.question{trial} = question; 
    
    % Load video   
    
    moviePtr = Screen('OpenMovie', win, path_stimulus);
    
    % Send trigger, video start
    if debug == false
        Datapixx('RegWrRd');
        data.trigTime{trial} = Datapixx('GetTime');
        switch condition
            case 'AO_U_XX'
                io64(ioObj, address, 10); %Pulse for MEG
            case 'AO_M_XX'
                io64(ioObj, address, 11); %Pulse for MEG
            case 'VO_U_BD'
                io64(ioObj, address, 20); %Pulse for MEG
            case 'VO_M_BD'
                io64(ioObj, address, 21); %Pulse for MEG
            case 'VO_X_SK'
                io64(ioObj, address, 22); %Pulse for MEG
            case 'VO_X_SC'
                io64(ioObj, address, 23); %Pulse for MEG
            case 'AV_U_BD'
                io64(ioObj, address, 30); %Pulse for MEG
            case 'AV_M_BD'
                io64(ioObj, address, 31); %Pulse for MEG
            case 'AV_X_SK'
                io64(ioObj, address, 32); %Pulse for MEG
            case 'AV_X_SC'
                io64(ioObj, address, 33); %Pulse for MEG   
        end
        WaitSecs(0.01);
        io64(ioObj, address, 0); 
    end
    
    % Play video
    Screen('PlayMovie', moviePtr, 1);
    toTime = inf;
    t = GetSecs();
    while t < toTime
        tex = Screen('GetMovieImage', win, moviePtr);
        if tex <= 0
            break;
        end
        
        Screen('DrawTexture', win, tex, [], rectVid);
        if strcmp(condition(1:2), 'AO')
            draw_fixation(win, white, xCenter, yCenter);
        else
            t = Screen(win, 'flip');
        end
        Screen('Close', tex);
    end
    Screen('PlayMovie', moviePtr, 0);
    Screen('CloseMovie', moviePtr);
    Screen(win, 'flip');
    
    % Send trigger, video end?
    if debug == false
        io64(ioObj, address, 99); %Pulse for MEG
        WaitSecs(0.01);
        io64(ioObj, address, 0);
    end
    
    WaitSecs(2);
    
    % Show question
    Screen('TextFont', win, 'Helvetica', 0);
    Screen('TextSize', win, 40);
    Screen('TextColor', win, [10 10 10]);
    DrawFormattedText(win, num2str(trial), 'left', 0.1*height);
    Screen('TextColor', win, [255 255 255]);
    DrawFormattedText(win, question.question, 'center', 0.3*height);
    DrawFormattedText(win, question.answers_rand{1}, 0.4*width, 0.5*height);
    DrawFormattedText(win, question.answers_rand{2}, 0.4*width, 0.6*height);
    DrawFormattedText(win, question.answers_rand{3}, 0.4*width, 0.7*height);
    DrawFormattedText(win, question.answers_rand{4}, 0.4*width, 0.8*height);
    Screen('FillRect', win , [255 0 0], [700*ratioW 497*ratioH 750*ratioW 547*ratioH]);
    Screen('FillRect', win , [255 255 0], [700*ratioW 605*ratioH 750*ratioW 655*ratioH]);
    Screen('FillRect', win , [0 255 0], [700*ratioW 713*ratioH 750*ratioW 763*ratioH]);
    Screen('FillRect', win , [0 127 255], [700*ratioW 821*ratioH 750*ratioW 871*ratioH]);
    Screen(win, 'flip');
    
    % Button press
    if debug == false
        [key,tim] = meg_KbWait_SN(Inf);
        Datapixx('RegWrRd');
    else
        [key,tim] = KbStrokeWait(Inf);
    end
    
    if debug == false
        data.trigTime{trial} = Datapixx('GetTime');
        switch key
            case 100 % Red
                io64(ioObj, address, 101);
            case 200 % Yellow
                io64(ioObj, address, 102);
            case 300 % Green
                io64(ioObj, address, 103);
            case 400 % Blue
                io64(ioObj, address, 104);
        end
        WaitSecs(0.010);
        io64(ioObj, address, 0);
    end
    
    data.resp{trial} = key;
    data.RT{trial} = tim;
    
    if debug == false
        switch key       
            case 100
                selected_answer = question.answers_rand{1};
                data.button_pressed{trial} = 'red';
            case 200
                selected_answer = question.answers_rand{2};
                data.button_pressed{trial} = 'yellow';
            case 300
                selected_answer = question.answers_rand{3};
                data.button_pressed{trial} = 'green';
            case 400
                selected_answer = question.answers_rand{4};
                data.button_pressed{trial} = 'blue';
        end
    else
        switch key       
            case 65 %A
                selected_answer = question.answers_rand{1};
                data.button_pressed{trial} = 'red';
            case 66 %B
                selected_answer = question.answers_rand{2};
                data.button_pressed{trial} = 'yellow';
            case 67 %C
                selected_answer = question.answers_rand{3};
                data.button_pressed{trial} = 'green';
            case 68 %D
                selected_answer = question.answers_rand{4};
                data.button_pressed{trial} = 'blue';
        end
    end
    
    data.selected_answer{trial} = selected_answer;
    
    % Check answer
    if strcmp(selected_answer, question.good_answer)
        Screen('TextFont', win, 'Helvetica', 0);
        Screen('TextSize', win, 50);
        DrawFormattedText(win, 'Correcto!', 'center', 'center');
    else
        Screen('TextFont', win, 'Helvetica', 0);
        Screen('TextSize', win, 50);
        DrawFormattedText(win, 'Falso!', 'center', 'center');    
    end
    Screen(win, 'flip');
    WaitSecs(2);
    Screen(win,'FillRect',black);
    Screen(win, 'flip');
    WaitSecs(2);
    
    % Break after 20 trials
    if ismember(trial, [21, 41, 61])
        Screen('TextFont', win, 'Helvetica', 0);
        Screen('TextSize', win, 50);
        DrawFormattedText(win, 'Ahora tienes una pausa para descansar.\n\nPulsa cualquier tecla cuándo quieres continuar.', 'center', 'center');        
        Screen(win, 'flip');
        if debug == false
            meg_KbWait_SN(Inf);
        else
            KbStrokeWait(Inf);
        end
        Screen(win,'FillRect',black);
        Screen(win, 'flip');
        WaitSecs(2);
    end

end

%% End screen
DrawFormattedText(win,'El experimento ha terminado! \n\n  Pulsa cualquier tecla para finalizar. \n\n Gracias :)','center','center', white);
Screen(win,'flip') 
if debug == false
    meg_KbWait_SN(Inf);
else
    KbStrokeWait(Inf);
end
WaitSecs(0.5);
Screen('CloseAll');

%%Save the results
filename = [path_results, subject];
save(filename, 'data');