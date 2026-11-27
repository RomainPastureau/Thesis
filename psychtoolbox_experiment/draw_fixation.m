%{
Name: draw_Fixation.m
Desc: Draws a fixation cross in the center of the screen
Authors: Mikel Lizarazu

Inputs:
- window: a Psychtoolbox window as given by Screen_Setup()
- white: white RGB values as given by Screen_Setup()
- xCenter: coordinates of the fixation cross center on the x axis as given
by Screen_Setup()
- yCenter: coordinates of the fixation cross center on the y axis as given
by Screen_Setup()
%}

function draw_fixation(window,white,xCenter,yCenter)
    % Draw the fixation cross in white, set it to the center of our screen
    % and set good quality antialiasing
    % Here we set the size of the arms of our fixation cross
fixCrossDimPix = 40;

% Now we set the coordinates (these are all relative to zero we will let
% the drawing routine center the cross in the center of our monitor for us)
xCoords = [-fixCrossDimPix fixCrossDimPix 0 0];
yCoords = [0 0 -fixCrossDimPix fixCrossDimPix];
allCoords = [xCoords; yCoords];

% Set the line width for our fixation cross
lineWidthPix = 4;
 % Draw the fixation cross in white, set it to the center of our screen
    % and set good quality antialiasing
    Screen('DrawLines', window, allCoords,lineWidthPix, white, [xCenter yCenter], 2);
    % Flip to the screen
    Screen('Flip', window);

end