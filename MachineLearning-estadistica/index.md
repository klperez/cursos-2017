---

title       : Machine Learning
subtitle    : Programa de Estadística  
author      : Kevin Pérez - Ing de Sistemas - Estadístico - (E) MSc. Ciencia de Datos  
job         : Departamento de Matemáticas y Estadística - Universidad de Córdoba
logo        : unicordoba3.png
framework   : io2012        # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : [mathjax, bootstrap, quiz, shiny, interactive]            
mode        : selfcontained # {standalone, draft}
ext_widgets : {rCharts: [libraries/nvd3]}
knit        : slidify::knit2slides
---

## Contenido programático 

**Unidad de aprendizaje Nº 1.** Generalidades 

> - Conceptos Básicos del **_Machine Learning_**.
> - Diseño de un estudio de predicción. 
> - Importancia relativa. 
> - Error en y fuera de la muestra. 
> - Tipos de errores. 
> - Validación cruzada y tecnicas de remuestreo. 
> - Preprocesamiento de los datos. 
> - Medidas de calidad de los modelos 

---

## Contenido programático 

**Unidad de aprendizaje Nº 2.**  _Machine Learning Supervisado_

> - Modelos de regresión 
>   - Regresión lineal simple y multiple 

> - Modelos de clasificación 
>   - Análisis discriminante lineal 
>   - Regresión logistica 

> - Métodos basados en Árboles 
>   - Árboles de decisión
>   - Bagging
>   - Random Forest 
>   - Boosting

---

## Contenido programático 

**Unidad de aprendizaje Nº 3.**  _Machine Learning No Supervisado_

> - Métodos de reducción de dimensionalidad 
>   - ACP, ACM, DVS

> - Métodos Cluster 
>   - K-Means 
>   - Cluster Jerárquico  

> - Reglas de asociación 

---

## Contenido programático 

**Unidad de aprendizaje Nº 4.**  Otros métodos en _Machine Learning_

> - _Optimización: Algoritmos Geneticos_

> - _Support Vector Machines_

> - _Neural Networks_

> - _Pronosticos: Series de tiempo_

---

## Contenido programático 

**Unidad de aprendizaje Nº 5.**  Optimización de los modelos 

> - _Tunning_

> - _Regulización en regresión_

> - _Combinación de modelos_

> - _Predicción basada en modelos_

---

## Referencias 

- Trevor H, Robert T, Jerome F,  _The Elements of Statistical Learning_, 2ª Edición, Springer.

- Gareth J, Daniela W, Trevor H, Robert T, _An Introduction to Statistical Learning with Applications in R_, 6ª Edición, Springer. 

- Ethem A, _Introduction to Machine Learning_, 2ª Edición, The MIT Press
Cambridge, Massachusetts.

- Max Kuhn, et all., _The caret package_, R CRAN, disponible en http://topepo.github.io/caret/index.html. 

---

## Motivación 

- Reconocido a nivel mundial en la academia

    - [Phd. in Machine Learning University of Alberta](https://www.ualberta.ca/computing-science/graduate-studies/programs-and-admissions/statistical-machine-learning)
    - [Phd. in Machine Learning Carnegie Mellon University](http://www.ml.cmu.edu/prospective-students/ml-phd.html)

- Alta demanda laboral 

    - [Statistical Jobs](http://hagutierrezro.blogspot.com.co/p/jobs.html)
    - [Data Scientist](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century)
    
- Un deporte moderno 

    - [Competencias](https://www.kaggle.com/competitions)
    - [Premios](http://www.netflixprize.com)


---


## Que es _Machine Learning_


> - Construcción/uso de algoritmos que _**aprenden**_ de los datos 

> - Más información implica mejor _**desempeño**_

> - Soluciones previas implican _**Experiencia**_ 

---


## Ejemplo 

- Etiquetar un cuadrado: Tamaño y borde  ---- Color 

- Previas observaciones (Etiquetadas por personas): 

<center>![](assets/img/img1.png)</center>

- Tarea de la maquina: _**Etiquetar**_ un nuevo cuadrado 

- Resultado: Éxito o Fracaso! 

---

## Formulación 



<center>![](assets/img/img2.png)</center>



---

## Que no es _Machine Learning_

- _**No**_ es machine learning 

    - Determinar el color que se presenta con mayor frecuencia 
    - Calcular el tamaño promedio del cuadrado 
    
- _**La Meta principal**_: Construir modelos para la predicción  

---

## Un problema de  Regresión 

<center>![](assets/img/img3.png)</center>

---

## Predicción 

<center>![](assets/img/img4.png)</center>

---

## _Statistical Learning_

El _Statistical Learning_ se refiere al vasto conjunto de herramientas para la _comprensión de los datos_. Estas herramientas pueden ser clasificadas _supervisadas_ o _no supervisadas_.

> - _supervised statistical learning:_ Implica la construcción de un modelo estadístico para predecir, o estimar, una _salida_ basada en una o más _entradas_.

> - _unsupervised statistical learning:_ Con estos modelos, existen _entradas_ pero no existen salidas supervisadas, sin embargo se puede aprender de la estructura de los datos.

---

## Por que estimar $f$

Existen dos razones principales por las cuales quisiéramos estimar $f$: _**Predicción**_ e _**Inferencia**_, teniendo en cuenta que: 

$$\mathbf{Y}= f(\mathbf{Y})+ \epsilon$$

Asumiendo las restricciones de cada modelo y teniendo en cuenta la naturaleza de cada unas las variables involucradas en el mismo.

---

## Predicción 

En muchas situaciones, un conjunto de _entradas_ $X$ se encuentran disponibles, pero la salida $Y$ no puede ser obtenida fácilmente. En esta situación y asumiendo que el termino de error es cero, podemos predecir $Y$ utilizando 

$$\hat{Y}=\hat{f}(X),$$
Donde $\hat{f}$ representa nuestra estimación para $f$ y $\hat{Y}$ representa la predicción resultante para $Y$. 

---

## Predicción 

Bajo estas condiciones $\hat{f}$ a menudo es considerada una _**caja negra**_ y en este sentido no se tiene en cuenta o nos interesa la forma de la función $\hat{f}$ siempre que de ella resulten buenas predicciones. 

<center>![](assets/img/img5.png)</center>


---

## Inferencia 

En este caso a menudo el interés se centra no solo en una buena predicción, también en la forma en que $Y$ se ve afectada por los cambios en $X_1, \ldots, X_p$. En otras palabras estamos interesados en la relación que guarden $X$ y $Y$ o en los cambios de $Y$ como función de $X_1, \ldots, X_p$. En este sentido es lógico tratar de responder las siguientes preguntas

> - ¿Qué predictores están asociados con la respuesta?
> - ¿Cuál es la relación entre la respuesta y cada predictor?
> - ¿Se puede resumir adecuadamente la relación entre $Y$ y cada predictor, 
usando una ecuación lineal, o es la relación más complicada?


---

## Como estimamos $f$

Hablando de una manera muy general y teniendo en cuenta que queremos encontrar una función $\hat{f}$ tal que $Y\approx \hat{f}(X)$ para cada observación $(X, Y)$, los métodos estadísticos para esta tarea pueden ser clasificados como: 

- Métodos Paramétricos 
- Métodos No Paramétricos 


---

## Métodos Paramétricos 

Los métodos paramétricos involucran un enfoque de dos pasos para el planteamiento del modelo

1. Se asumen unos supuestos acerca de la forma funcional de $f$, por ejemplo, un supuesto muy simple es que $f$ es lineal en $X$:

$$f(X)= \beta_0+\beta_1X_1+\beta_2X_2+\cdots +\beta_pX_p$$
2. Una vez el modelo fue seleccionado, necesitamos un procedimiento de ajuste. En el caso del modelo lineal necesitamos estimar los parámetros $\beta_0, \beta_1, \ldots, \beta_p$, esto es, encontrar los valores de esos parámetros tal que 

$$Y\approx \beta_0+\beta_1X_1+\beta_2X_2+\cdots +\beta_pX_p$$

---

## Métodos Paramétricos

<center>![](assets/img/img6.png)</center>

---

## Métodos No Paramétricos 

Los métodos no paramétricos hacen o lanzan supuestos explícitos acerca de la forma funcional de $f$, en lugar de eso buscan una estimación de $f$ que se aproxima a los puntos de los datos como sea posible sin ser demasiado áspera o sinuosa (ondulada).

<center>![](assets/img/img5.png)</center>

---

## Compensación entre flexibilidad e interpretabilidad 

Los métodos estadísticos del _machine learning_ propuestos anteriormente algunos son menos flexibles o menos restrictivos, en el sentido de que pueden producir sólo una gama relativamente pequeña de formas para estimar $f$, otros métodos como el _thin plate splines_ son mucho más flexibles porque pueden generar una gama mucho más amplia de formas posibles para estimar $f$

<center>![](assets/img/img7.png)</center>

---
