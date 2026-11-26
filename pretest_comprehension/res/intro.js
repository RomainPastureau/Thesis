var timelineIntro = [];
var timelineSoundCheck = [];
var timelineInstructions1 = [];

var welcome = {
	type: jsPsychHtmlButtonResponse,
	stimulus: '<p><strong><span style="font-size:26px;">Presentación: Proyecto BLComp</span></strong></p><table align="center" width=60%>'+
	'<tr><td width=80%><p style="text-align:right;"><em>Basque Center on Cognition, Brain and Language es un centro '+
	'internacional de investigación interdisciplinar para el estudio de la cognición, el cerebro y el lenguaje. '+
	'Visita nuestra <a href="http://www.bcbl.eu">página web</a> (www.bcbl.eu) y síguenos en redes sociales como '+
	'<a href="https://www.facebook.com/bcbl.eu">Facebook</a> (@bcbl.eu), <a href="https://www.instagram.com/bcbl_'+
	'basque_center/">Instagram</a> (bcbl_basque_center), <a href="https://twitter.com/bcbl_">Twitter</a> (@bcbl_) y'+
	' <a href="https://www.youtube.com/user/BCBLchannel">YouTube</a> (BCBL) para conocer todas las novedades de nuestro '+
	'centro así como del campo de la neurociencia cognitiva.</em></p></td><td><p><span style="font-size:14px;"><span '+
	'style="font-family:arial,helvetica,sans-serif;"><em><img alt="" src="https://www.bcbl.eu/participa/img/logobcbl.png" '+
	'style="width: 197px; height: 163px; float: right;" /></em></span></span></p></td></tr></table>',
	choices: ['Continuar'],
};
timelineIntro.push(welcome);

/* get participant ID */
var id_participa = {
    type: jsPsychSurveyText,
    questions: [{prompt: 'Por favor escribe tu código de participa. Luego pulsa «Continuar».', "required": true, "placeholder": 'Código de participa BCBL (max. 5 numeros)', "columns": 4,
    "minlength":2, "maxlength":5}],
    button_label: "Continuar",
};
timelineIntro.push(id_participa);

/* get participant DNO */
var id_DNI = {
    type: jsPsychSurveyText,
    questions: [{prompt: 'Por favor escribe tu DNI/NIE. Luego pulsa «Continuar».', "required": true,
    "placeholder":"DNI/NIE", "minlength":9, "maxlength":9}],
    button_label: "Continuar",
};
timelineIntro.push(id_DNI);

/* get participant DNI */
var id_name = {
    type: jsPsychSurveyText,
    questions: [{prompt: 'Por favor escribe tu nombre y apellidos. Luego pulsa «Continuar».', "required": true,
    "placeholder":"Nombre"}, {prompt: "", "required": true, "placeholder":"Apellidos"}],
    button_label: "Continuar",
};
timelineIntro.push(id_name);

var option_ic = ["He leído y acepto el consentimiento informado y autorizo el tratamiento de mis datos de carácter personal."];

var informed_consent = {
	type: jsPsychSurveyMultiChoice,
	questions: [{prompt: "<p><b>CONSENTIMIENTO INFORMADO:</b></p>"+
	"<p>La participación en este estudio es totalmente voluntaria y el/la participante recibirá compensación económica. "+
    "Este estudio tiene únicamente fines de investigación y los datos recogidos no se utilizarán con fines diagnósticos o clínicos. "+
    "La utilización de los datos de los/las participantes para el estudio final dependerá en última instancia de los investigadores "+
    "responsables, pudiendo no ser incluidos por motivos técnicos u otras circunstancias concretas. "+
    "Ningún dato publicado como consecuencia del estudio revelará la identidad del/de la participante.</p>"+
    "<p><b>AUTORIZACIÓN PARA EL TRATAMIENTO DE LOS DATOS DE CARÁCTER PERSONAL:</b></p>"+
	"<p>Sus datos de carácter personal, o de la persona a la que representa, serán tratados y se incorporarán "+
	"a ficheros de los que es responsable el BCBL y que tienen como finalidad la gestión y la administración del centro, "+
	"la realización de estudios sobre la cognición, el cerebro y el lenguaje y el envío de comunicaciones relacionadas con estos "+
	"fines por cualquier medio. La legitimación para el tratamiento es el consentimiento de la persona interesada. "+
	"Estos datos, que se conservarán de forma indefinida, solo podrán ser cedidos, anonimizados, a instituciones o personas dedicadas "+
	"a fines análogos, salvo las cesiones legalmente establecidas.</p>"+
	"<p>Podrá ejercitar los derechos de acceso, rectificación, supresión, portabilidad, limitación de tratamiento, "+
	"revocar el consentimiento o no ser objeto de decisiones basadas únicamente en el tratamiento automatizado de sus datos, "+
	"de conformidad al Reglamento (UE) 2016/679 y a la ley Orgánica 3/2018 de 5 de diciembre de protección de datos personales "+
	"y garantía de los derechos digitales dirigiendo un correo electrónico a <a href='mailto:info@bcbl.eu'><b>info@bcbl.eu</b></a> o bien mediante escrito dirigido al "+
	"Delegado de Protección de Datos a la dirección Paseo Mikeletegi, 69 – 2º - 20009 SAN SEBASTIÁN, acompañando fotocopia de su DNI.</p>"+
	"<p>* Para más información sobre la política de privacidad del BCBL pincha en <a href='https://www.bcbl.eu/es/politica-privacidad'>Política de Privacidad</a>.</p>", 
	options: option_ic, required: true, button_label: "Continuar"}],
};
timelineIntro.push(informed_consent);

var option_exp = ["Validar"];

var info_experiment = {
	type: jsPsychSurveyMultiChoice,
	questions: [{prompt: "<p>¡Gracias por participar en este estudio! Por favor, lee detenidamente la siguiente información y no dudes en "+
	"preguntarnos si precisas alguna aclaración adicional. Muchas gracias.</p>"+
	"<p><b>¿Cuál es nuestro objetivo con este estudio?</b></p>"+
	"<p>Vas a escuchar 72 fragmentos de audio de quince segundos de duración en los que "+
	"se oye a personas contando lo que recuerdan sobre un video de dibujos animados que acaban de ver.</p>"+
	"<p>Tras cada fragmento, deberás indicar si, en base a lo que has oído, el hablante llevaba mascarilla o no mientras hablaba.</p>"+
	"<p>Utiliza el ratón para seleccionar tu respuesta (“Lleva mascarilla” o “No lleva mascarilla”). Por favor, utiliza auriculares en "+
	" un ambiente sin ruido para llevar a cabo este experimento. </p>"+
	"<p><b>Informaciones complementarias</b></p>"+
	"<p>Este estudio durará aproximadamente 20 minutos. Es preferible pasar este estudio en tu ordenador, y no en tu móvil o tableta.</p>"+
	"<p>Para cualquier pregunta científica referente a este estudio, por favor no dudes en enviar un email al investigador "+
	"indicando el nombre del proyecto (BLComp) y el nombre del estudio. "+
	"Para cualquier otro aspecto ténico, por favor escribe un email a participa@bcbl.eu o llámanos al 943.309.300.</p>"+
	"<p>La compensación económica por participar en este estudio asciende a 8 euros. Para recibirla, "+
	"deberás completar la prueba hasta el final, siguiendo estrictamente las instrucciones indicadas. Una vez completes "+
	"el estudio y comprobemos los datos, nos pondremos en contacto contigo para abonarte la cantidad correspondiente.</p>"+
	"<p>Por favor, una vez comenzado el test, es importante realizar el estudio todo seguido hasta el final y evitando distracciones "+
	"(móvil, ruidos, levantarse, etc). Intenta hacerlo lo mejor posible ya que, una vez recibidos todos los resultados, un comité de "+
	"investigadores chequeará todos los datos para evitar fraudulencias.</p>", 
	options: option_exp, required: true, button_label: "Continuar"}],
};
timelineIntro.push(info_experiment);

var introduction = {
  timeline: timelineIntro,
}

var i;
for (i = 10; i > 0; i--) {
	var instructions = {
		type: jsPsychHtmlKeyboardResponse,
		stimulus: "<p>Antes de empezar, vamos a verificar si puedes oír los audios.</p>"+
		"<p>A continuación, vas a escuchar un sonido.</p>"+
		"<p>Por favor, utiliza auriculares en un ambiente sin ruido.</p>"+
		"<p>Haz clic en 'Sí' si puedes oír el sonido. Si no puedes, verifica que tu sistema "+
		"de sonido funciona correctamente, y haz clic en 'Intentar de nuevo'</p>"+
		'<p style="color:#888888">Pulsa la BARRA ESPACIADORA para empezar.</i></p>'+
		"<p><b>"+i.toString()+"</b></p>",
		choices: "NO_KEYS",
		trial_duration: 1000,
	};
	timelineSoundCheck.push(instructions);
}

var instructions = {
	type: jsPsychHtmlKeyboardResponse,
	stimulus: "<p>Antes de empezar, vamos a verificar si puedes oír los audios.</p>"+
	"<p>A continuación, vas a escuchar un sonido.</p>"+
	"<p>Por favor, utiliza auriculares en un ambiente sin ruido.</p>"+
	"<p>Haz clic en 'Sí' si puedes oír el sonido. Si no puedes, verifica que tu sistema "+
	"de sonido funciona correctamente, y haz clic en 'Intentar de nuevo'</p>"+
	'<p style="color:#ff0000"><b>Pulsa la BARRA ESPACIADORA para empezar.</b></p><p>&nbsp;</p>',
	choices: [" "], //return key
};
timelineSoundCheck.push(instructions);

var soundcheckinstructions = {
  timeline: timelineSoundCheck,
}

var i;
for (i = 10; i > 0; i--) {
	var instructions = {
		type: jsPsychHtmlKeyboardResponse,
		stimulus: "<p>A continuación, vas a escuchar 72 fragmentos de audio de quince segundos de duración</p>"+
		"<p>en los que se oye a personas contando lo que recuerdan sobre un video de dibujos animados que acaban de ver.</p>"+
		"<p>Tras cada fragmento, deberás indicar si, en base a lo que has oído, el hablante llevaba mascarilla o no mientras hablaba.</p>"+
		"<p>Utiliza el ratón para seleccionar tu respuesta (“Lleva mascarilla” o “No lleva mascarilla”).</p>"+
		"<p>Por favor, utiliza auriculares en un ambiente sin ruido para llevar a cabo este experimento.</p>"+
		"<p>Puedes hacer una pausa cuando quieras durante el experimento.</p>"+
		'<p style="color:#888888">Pulsa la BARRA ESPACIADORA para empezar.</i></p>'+
		"<p><b>"+i.toString()+"</b></p>",
		choices: "NO_KEYS",
		trial_duration: 1000,
	};
	timelineInstructions1.push(instructions);
}

var instructions = {
	type: jsPsychHtmlKeyboardResponse,
		stimulus: "<p>A continuación, vas a escuchar 72 fragmentos de audio de quince segundos de duración</p>"+
		"<p>en los que se oye a personas contando lo que recuerdan sobre un video de dibujos animados que acaban de ver.</p>"+
		"<p>Tras cada fragmento, deberás indicar si, en base a lo que has oído, el hablante llevaba mascarilla o no mientras hablaba.</p>"+
		"<p>Utiliza el ratón para seleccionar tu respuesta (“Lleva mascarilla” o “No lleva mascarilla”).</p>"+
		"<p>Por favor, utiliza auriculares en un ambiente sin ruido para llevar a cabo este experimento.</p>"+
		"<p>Puedes hacer una pausa cuando quieras durante el experimento.</p>"+
		'<p style="color:#ff0000"><b>Pulsa la BARRA ESPACIADORA para empezar.</b></p><p>&nbsp;</p>',
	choices: [" "], //return key
};
timelineInstructions1.push(instructions);

var instructions1 = {
  timeline: timelineInstructions1,
}