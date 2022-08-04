
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class Adaline {
	
	//Funcion que lee ficheros
	private static List<String> readFile(String path) throws IOException {
		Path ruta = Paths.get(path);
		return Files.readAllLines(ruta, StandardCharsets.UTF_8);
	}
	
	//Funcion que crea ficheros
	public static void escribir(String[] newFile,String path) throws IOException{
		List<String> lineas = Arrays.asList(newFile);
		Path file = Paths.get(path);
		Files.write(file, lineas, Charset.forName("UTF-8"));
	}

	public static void main(String[] args) throws IOException {
		
		System.out.println("...");
		int n_atributos = 9; //numero de atributos incluyendo el de la salida.
		
		// Fichero de entrenamiento. Se lee y se crea una matriz con tantas filas y columnas como lineas de entrenamiento y atributos haya.
		List<String> lineas_entrenamiento = readFile("D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\Entrenamiento.txt");
		Double[][]entrenamiento = new Double[lineas_entrenamiento.size()][n_atributos];

		// Fichero de evaluacion. Se lee y se crea una matriz con tantas filas y columnas como lineas de entrenamiento y atributos haya.
		List<String> lineas_evaluacion = readFile("D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\Validacion.txt");
		Double[][]validacion = new Double[lineas_evaluacion.size()][n_atributos];

		// Fichero de test. Se lee y se crea una matriz con tantas filas y columnas como lineas de entrenamiento y atributos haya.
		List<String> lineas_test = readFile("D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\Test.txt");
		Double[][] test = new Double[lineas_test.size()][n_atributos];		
		
//---------------------------SE RELLENAN LAS MATRICES CREADAS CON LOS DATOS LEIDOS CONVIRTIENDOLOS A DOUBLE--------------------------------------	
		
		//matriz de entrenamiento
		for (int i = 0; i < lineas_entrenamiento.size(); i++) {
			String[] aux = lineas_entrenamiento.get(i).split(";");
			for (int j = 0; j < n_atributos; j++) {
				entrenamiento[i][j] = Double.parseDouble(aux[j]);
			}
		}
		//matriz de validación
		for (int i = 0; i < lineas_evaluacion.size(); i++) {
			String[] aux = lineas_evaluacion.get(i).split(";");
			for (int j = 0; j < n_atributos; j++) {
				validacion[i][j] = Double.parseDouble(aux[j]);
			}
		}
		//matriz de test
		for (int i = 0; i < lineas_test.size(); i++) {
			String[] aux = lineas_test.get(i).split(";");
			for (int j = 0; j < n_atributos; j++) {
				test[i][j] = Double.parseDouble(aux[j]);
			}
		}
			
//----------------------------SE INICIALIZAN VARIABLES PARA APRENDIZAJE-----------------------------------
		
		int num_ciclos = 2958;  //Criterio de parada: cuando pasen numero de ciclos necesarios para minimo error de validacion = 2958 con 0.001
		double tasa_de_aprendizaje = 0.001;
		//Inicializacion de pesos y umbral aleatorios.
		int n_pesos = (n_atributos - 1);
		Double[] pesos = new Double[n_pesos];
		
		/*for (int i = 0; i < n_pesos; i++) {
			pesos[i] = Math.random()*2-1;
			System.out.println("El peso " + i + " es: " + pesos[i]); //Usado para sacar los pesos aleatorios
		  }*/
		
		pesos[0] = -0.8474419824483137;
		pesos[1] = 0.6908746374529446;
		pesos[2] = 0.29603617106066493;
		pesos[3] = 0.16535669707662004;
		pesos[4] = -0.8457962892753188;
		pesos[5] = -0.8912057552176695;
		pesos[6] = -0.014293072963992515;
		pesos[7] = -0.6078303273360675;
																		
		/*double umbral = Math.random()*2-1;
		System.out.println("El umbral es: " + umbral);*/ //Usado para sacar un umbral aleatorio.
		
		double umbral = 0.4343276439249215; 
		//Arrays para guardar los errores obtenidos tras cada ciclo
		String[] error_entrenamiento= new String[num_ciclos];
		String[] error_validacion= new String[num_ciclos];
		
		double error_cuadratico_medio_entrenamiento = 0;
		double error_cuadratico_medio_validacion = 0;
		double error_cuadratico_medio_test = 0;
		
		Double multiplica[] = new Double[n_pesos]; //Array donde se calculara la funcion de activacion de la neurona resultante de hacer (entrada*peso)+umbral
		//Variables auxiliares para guardar el momento a lo largo de los ciclos en que el error es minimo
		int n_ciclos_min = 0; 
		double error_validacion_min = 0.0;
		Double[] pesos_min = new Double[n_pesos];
		double umbral_min = 0.0;
		
//-------------------------------------------COMIENZO DEL APRENDIZAJE-------------------------------------------------		
		
		for (int q = 0; q < num_ciclos; q++) {
			//reinicializacion de las variables de error despues de cada ciclo
			error_cuadratico_medio_entrenamiento = 0;
			error_cuadratico_medio_validacion = 0;
		
			//se pasa el entrenamiento, cambiando los pesos en cada patron
			for (int i = 0; i < lineas_entrenamiento.size(); i++) {
				for (int j = 0; j < n_pesos; j++){
					multiplica[j] = entrenamiento[i][j] * pesos[j];
				}
				//se suma todo incluyedo el umbral
				double suma = 0;
				for (int j = 0; j < n_pesos; j++) {
					suma = suma + multiplica[j];
				}
				double y = suma + umbral;
				
				//se calcula la diferencia entre la salida deseada y la obtenida
				double diferencia = entrenamiento[i][n_atributos-1] - y;
				
				//se cambian los pesos
				for (int j = 0; j < n_pesos; j++) {
					pesos[j] = tasa_de_aprendizaje * (diferencia) * entrenamiento[i][j] + pesos[j];
				}
				umbral = tasa_de_aprendizaje * (diferencia) + umbral;
			}
			//se pasa el entrenamiento pero sin cambiar los pesos para calcular el error
			for (int i = 0; i < lineas_entrenamiento.size(); i++) {
				for (int j = 0; j < n_pesos; j++){
					multiplica[j] = entrenamiento[i][j] * pesos[j];
				}
				//se suma todo incluyedo el umbral
				double suma = 0;
				for (int j = 0; j < multiplica.length; j++) {
					suma = suma + multiplica[j];
				}
				double y = suma + umbral;
				//se calcula la diferencia entre la salida deseada y la obtenida
				double diferencia = entrenamiento[i][n_atributos-1] - y;
				
				error_cuadratico_medio_entrenamiento = error_cuadratico_medio_entrenamiento + Math.pow(diferencia, 2);
				
			}
			error_cuadratico_medio_entrenamiento = error_cuadratico_medio_entrenamiento/lineas_entrenamiento.size();
	
			//se pasa la matriz de evaluacion sin cambiar los pesos para calcular el error
			for (int i = 0; i < lineas_evaluacion.size(); i++) {
				for (int j = 0; j < n_pesos; j++){
					multiplica[j] = validacion[i][j] * pesos[j];
				}
				// Sumamos todo incluyedo el umbral
				double suma = 0;
				for (int j = 0; j < multiplica.length; j++) {
					suma = suma + multiplica[j];
				}
				double y = suma + umbral;
				//se calcula la diferencia entre la salida deseada y la obtenida
				double diferencia = validacion[i][n_atributos-1] - y;
				error_cuadratico_medio_validacion = error_cuadratico_medio_validacion + Math.pow(diferencia, 2);
			} 
			
			error_cuadratico_medio_validacion = error_cuadratico_medio_validacion/lineas_evaluacion.size();
			error_entrenamiento[q]=String.valueOf(error_cuadratico_medio_entrenamiento);
			error_validacion[q]=String.valueOf(error_cuadratico_medio_validacion);	
			
			//se comprueba si el error de validación es menor que el error de validacion del ciclo anterior. Si es asi guardamos la instancia de ese ciclo
			if(q==0) {
				error_validacion_min = error_cuadratico_medio_validacion;
				n_ciclos_min = q;
				for (int t = 0; t < n_pesos; t++) {
					pesos_min[t] = pesos[t];
				}
				umbral_min = umbral;
			}
			else if (error_cuadratico_medio_validacion < error_validacion_min) {
				error_validacion_min = error_cuadratico_medio_validacion;
				n_ciclos_min = q;
				for (int t = 0; t < n_pesos; t++) {
					pesos_min[t] = pesos[t];
				}	
				umbral_min = umbral;
			}
			
		}
	
//---------------------------------------SE PASA EL TEST-------------------------------------------------
	
//se pasa el test sin cambiar los pesos y se calcula el error.
		String[] salida_test= new String[lineas_test.size()];
		for (int i = 0; i < lineas_test.size(); i++) {
			for (int j = 0; j < n_pesos; j++){
				multiplica[j] = test[i][j] * pesos_min[j];
			}
			//se suma todo incluyedo el umbral
			double suma = 0;
			for (int j = 0; j < multiplica.length; j++) {
				suma = suma + multiplica[j];
			}
			double y = suma + umbral_min;
			salida_test[i]=String.valueOf(y);
			//se calcula la diferencia entre la salida deseada y la obtenida
			double diferencia = test[i][n_atributos-1] - y;
			error_cuadratico_medio_test = error_cuadratico_medio_test + Math.pow(diferencia, 2);
		}
		error_cuadratico_medio_test = error_cuadratico_medio_test/lineas_test.size();
		

//----------------SE ESCRIBEN ULTIMOS LOS PESOS Y UMBRAL EN UN ARRAY, ADEMAS DE: EL ULTIMO ERROR DE ENTRENAMIENTO, VALIDACION Y TEST---
		
		//se escriben los pesos en un array
		String[] pesos_umbral= new String[(pesos.length)+1];
		for (int i = 0; i < pesos_umbral.length-1; i++) {
			pesos_umbral[i]=String.valueOf(pesos[i]);
		}
		pesos_umbral[pesos_umbral.length-1]=String.valueOf(umbral);
		
		String[] error_todo= new String[3];
		error_todo[0]=String.valueOf(error_cuadratico_medio_entrenamiento);
		error_todo[1]=String.valueOf(error_cuadratico_medio_validacion);
		error_todo[2]=String.valueOf(error_cuadratico_medio_test);
		
	/*	for (int i = 0; i < pesos.length; i++) {
			System.out.println("El peso "+i+" es: "+pesos[i]);				
		}
		System.out.println("El umbral es "+umbral); */
		
		System.out.println("El error minimo de validacion es: "+ error_validacion_min);
		System.out.println("El ciclo en el que se ha conseguido dicho error es: "+ (n_ciclos_min+1));
		System.out.println("los pesos en este ciclo fueron:");
		for (int i = 0; i < pesos.length; i++) {
			System.out.println("El peso "+i+" fue: "+pesos_min[i]);				
		}
		System.out.println("El umbral fue "+umbral_min);
		
		escribir(salida_test,"D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\salida_test_" + tasa_de_aprendizaje + ".txt");
		escribir(error_todo,"D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\error_todo_" + tasa_de_aprendizaje + ".txt");
		escribir(pesos_umbral,"D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\pesos_umbral_" + tasa_de_aprendizaje + ".txt");
		escribir(error_entrenamiento,"D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\error_medio_entrenamiento_" + tasa_de_aprendizaje + ".txt");
		escribir(error_validacion,"D:\\GRADO\\Redes de Neuronas Artificiales\\Practica1\\Practica1_Adaline\\error_medio_validacion_" + tasa_de_aprendizaje + ".txt");
	}//main
	
}//class