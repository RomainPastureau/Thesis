var timelineOutro = [];

var bye = {
	type: jsPsychHtmlKeyboardResponse,
	stimulus: '<p style="text-align:justify;"><strong><span style="font-size:26px;">Muchas gracias por participar en nuestro estudio.</span></strong></p>'+
	'<p style="text-align:justify;">Una vez hayamos recibido todos los resultados necesarios para acabar el estudio, chequearemos todos los datos y nos pondremos '+
	'en contacto contigo para realizar el pago. Mientras tanto, puedes leer el siguiente debriefing donde podrás entender cuáles eran '+
	'las hipótesis de este estudio y que no hemos podido explicarte antes de comenzar el estudio para que tus respuestas no estuviesen '+
	'condicionadas por esta información. ¡Esperamos que te sea de interés!</p>'+
	'<p style="text-align:justify;">¡Estate atento/a a los medios habituales (tu perfil de BCBL, e-mail, etc.) para participar en nuestros experimentos en línea y '+
	'visita nuestra <a href="http://www.bcbl.eu">página web</a> (www.bcbl.eu) y síguenos en redes sociales como '+
	'<a href="https://www.facebook.com/bcbl.eu">Facebook</a> (@bcbl.eu), <a href="https://www.instagram.com/bcbl_'+
	'basque_center/">Instagram</a> (bcbl_basque_center), <a href="https://twitter.com/bcbl_">Twitter</a> (@bcbl_) y'+
	' <a href="https://www.youtube.com/user/BCBLchannel">YouTube</a> (BCBL) para conocer todas las novedades de nuestro '+
	'centro así como del campo de la neurociencia cognitiva.'+
	'<p style="text-align:justify;"><strong><span style="font-size:26px;">Debriefing</span></strong></p><p style="text-align:justify;">'+
	"¡Gracias por participar en este experimento! Su objetivo es comprobar si el hecho de que el hablante lleve mascarilla o "+
	"no se puede detectar conscientemente. Se sabe que llevar mascarilla modifica algunas de las propiedades acústicas del habla; "+
	"en relación a esto, has estuchado audios de personas que llevaban mascarilla, otros en los que no llevaban mascarilla y otros "+
	"en los que no llevaban mascarilla y con un tipo de procesamiento que imita la forma en la que suena el habla cuando alguien "+
	"lleva mascarilla. Queremos averiguar si el procesamiento del audio de esta manera hace que los fragmentos de audio con y "+
	"sin mascarilla resulten indistinguibles.</p>"+
	'<p>Una vez nos hayamos asegurado de que has realizado las tareas correctamente, nos pondremos en contacto contigo vía email para realizar el pago.</p>'+
	'<p style="text-align:center;"><b>Ya puedes cerrar la pantalla o pulsar la tecla RETURN.</b></p>',
	choices: ["Enter"],
};
timelineOutro.push(bye);

var outro = {
  timeline: timelineOutro,
}