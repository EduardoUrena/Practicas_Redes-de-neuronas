library(RSNNS)


## funcion que calcula el error cuadratico medio
MSE <- function(pred,obs) {sum((pred-obs)^2)/length(obs)}


#CARGA DE DATOS
# se supone que los ficheros tienen encabezados
 trainSet <- read.csv("Train.csv",dec=".",sep=",",header = T)
 validSet <- read.csv( "Valid.csv",dec=".",sep=",",header = T)
 testSet  <- read.csv("Test.csv",dec=".",sep=",",header = T)

 #trainSet <- read.table("trainParab.dat")
 #validSet <- read.table( "testParab.dat")
 #testSet <- read.table( "testParab.dat")


salida <- ncol (trainSet)   #num de la columna de salida





#SELECCION DE LOS PARAMETROS
topologia        <- 100 #PARAMETRO DEL TIPO c(A,B,C,...,X) A SIENDO LAS NEURONAS EN LA CAPA OCULTA 1, B LA CAPA 2 ...
razonAprendizaje <- 0.2 #NUMERO REAL ENTRE 0 y 1
ciclosMaximos    <- 10000 #NUMERO ENTERO MAYOR QUE 0

#EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO

set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
             )

#GRAFICO DE LA EVOLUCION DEL ERROR
plotIterativeError(model)

# DATAFRAME CON LOS ERRORES POR CICLo: de entrenamiento y de validacion
iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MSEValid= (model$IterativeTestError/nrow(validSet)))

######################################################################
#SE OBTIENE EL NuMERO DE CICLOS DONDE EL ERROR DE VALIDACION ES MINIMO
#######################################################################

nuevosCiclos <- which.min(model$IterativeTestError)

#ENTRENAMOS LA MISMA RED CON LAS ITERACIONES QUE GENERAN MENOR ERROR DE VALIDACION
set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=nuevosCiclos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
)
#GRAFICO DE LA EVOLUCION DEL ERROR
plotIterativeError(model)

iterativeErrors1 <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MSEValid= (model$IterativeTestError/nrow(validSet)))

#CALCULO DE PREDICCIONES
prediccionesTrain <- predict(model,trainSet[,-salida])
prediccionesValid <- predict(model,validSet[,-salida])
prediccionesTest  <- predict(model, testSet[,-salida])

#CALCULO DE LOS ERRORES
errors <- c(TrainMSE= MSE(pred= prediccionesTrain,obs= trainSet[,salida]),
            ValidMSE= MSE(pred= prediccionesValid,obs= validSet[,salida]),
            TestMSE=  MSE(pred= prediccionesTest ,obs=  testSet[,salida]))
errors





#SALIDAS DE LA RED
outputsTrain <- data.frame(pred= prediccionesTrain,obs= trainSet[,salida])
outputsValid <- data.frame(pred= prediccionesValid,obs= validSet[,salida])
outputsTest  <- data.frame(pred= prediccionesTest, obs=  testSet[,salida])




#GUARDANDO RESULTADOS
saveRDS(model,"nnet_100n_0.2.rds")
write.csv2(errors,"finalErrors_100n_0.2.csv")
write.csv2(iterativeErrors,"iterativeErrors_100n_0.2.csv")
write.csv2(outputsTrain,"netOutputsTrain_100n_0.2.csv")
write.csv2(outputsValid,"netOutputsValid_100n_0.2.csv")
write.csv2(outputsTest, "netOutputsTest_100n_0.2.csv")




# #############
# colnames(trainSet)=c("x","y","z")
# head(trainSet)
# modelo=lm(z~x+y, trainSet)
# 
# summary(modelo)
# mselin <- mean(modelo$residuals^2)
# mselin   #error mse
# 
# Fuente: https://www.i-ciencias.com/pregunta/89240/como-obtener-el-valor-del-error-cuadratico-medio-de-una-regresion-lineal-en-r
