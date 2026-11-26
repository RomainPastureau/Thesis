var trial = {
	type: jsPsychAudioButtonResponse,
	//stimulus: "sounds/001.mp3",
	stimulus: jsPsych.timelineVariable('stimulus'),
	id: jsPsych.timelineVariable('id'),
	choices: ["Lleva mascarilla", "No lleva mascarilla"],
	prompt: "¿El hablante tiene una mascarilla?",
	response_allowed_while_playing: false,
	data: jsPsych.timelineVariable('data'),
}

var experiment_loop = {
	timeline: [trial],
	timeline_variables: stimuli_experiment,
	randomize_order: true
};
