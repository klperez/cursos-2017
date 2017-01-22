---

title       : Bioestadística 
subtitle    : Programa de Biología  
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

**Unidad de aprendizaje Nº 1.** Estadística descriptiva 

> - Conceptos Básicos.
> - Organización de datos.
> - Medidas de tendencia central, posición y dispersión.
> - Representaciones gráficas y análisis exploratorio de datos (AED)
> - Aplicaciones en el campo de la Biología. 

---

## Contenido programático 

**Unidad de aprendizaje Nº 2.** Teoría de probabilidad

> - Elementos de la teroría de la probabilidad.
> - Probabilidad condicional, independencia y teorema de Bayes. 
> - Variables aleatorias.
> - Distribuciones de probabilidad, discretas y continuas.
> - Aplicaciones en el campo de la Biología. 

---

## Contenido programático 

**Unidad de aprendizaje Nº 3.** Distribuciones muestrales

> - Distribuciones muestrales con uno o más parámetros
> - Estimación puntual 
> - Aplicaciones en el campo de la Biología. 

---

## Contenido programático 

**Unidad de aprendizaje Nº 4.** Estimación de parámetros y pruebas de hipótesis

> - Estimación por intervalos de confianza 
> - Pruebas de hipótesis en una población.
> - Pruebas de hipótesis en dos poblaciones.
> - Pruebas de hipótesis no paramétricas.
> - Aplicaciones en el campo de la Biología. 

--- 

## Referencias 
> - Estadística para Biología y Ciencias de la salud, J Susan Milton, 3.ª Edición, McGraw Hill. 

> - Curso elemental de Estadística y Pobabilidad, Luis Rincon, 1.ª Edición, UNAM.

> - Introducción a R, R Development Core Team, Versión 1.0.1 (2000-05-16), CRAN, 
https://cran.r-project.org/doc/contrib/R-intro-1.1.0-espanol.1.pdf

---


## Conceptos Generales  

- ¿Qué es Estadística?

---

## Conceptos Generales 

La _**Estadística**_ se ocupa del manejo de la información que pueda ser cuantificada. Implica esto la descripción de con juntos de datos y la inferencia a partir de la información recolectada de un fenómeno de interés. La función principal de la estadística abarca: Resumir, Simplificar, Comparar, Relacionar, Proyectar. Entre las tareas que debe enfrentar un estudio estadístico están:

> - Delimitar con precisión la población de referencia o el conjunto de datos en estudio, las unidades que deben ser observadas, las características o variables que serán medidas u observadas.

> - Estrategias de Observación: Censo, Muestreo, Diseño de Experimental

> - Recolección y Registro de la información


---

## Conceptos Generales 

>  -  Depuración de la información.

>  - Construcción de Tablas.

>  - Análisis Estadístico:
      * Producción de resúmenes gráficos y numéricos.
      * Interpretación de resultados.

>  Cuando los datos comprenden toda la población de referencia, hablamos de un Censo y cuando solo comprometen una parte de ella, hablamos de una muestra. En ambos casos es pertinente un análisis Descriptivo. En el segundo caso un análisis Inferencial.      

---  

## Conceptos Generales 

A grandes rasgos podemos decir que una _**Población**_ es el conjunto de toda posible información, o de los objetos, que permite estudiar un fenómeno de interés. 

> - Una _**muestra**_ es un subconjunto de información representativa de una población.

> - Las _**Variables**_ resultan ser aquellas características de interés que desean ser medidas sobre los objetos o individuos seleccionados. En la mayoría de los casos lo que se pretende es estimar, a partir de la información recolectada de una muestra, características desconocidas de los objetos en dicha población de interés.

> - Las características desconocidas de una población serán llamadas _**Parámetros**_. Las características calculadas a partir de una muestra son llamadas _**Estadísticas**_. Una Inferencia es una generalización obtenida a partir de una muestra aleatoria.

---

## Conceptos Generales 

La Estadística puede dividirse en dos grandes ramas: Estadística Descriptiva y Estadística Inferencial.

> - _**Estadística descriptiva:**_ Es el conjunto de métodos usados para la organización y presentación (descripción) de la información recolectada. La información recolectada puede ser catalogada de dos maneras: Datos Cualitativos y Cuantitativos.

> - _**Estadística inferencial:**_ Comprende los métodos y procedimientos para deducir propiedades (hacer inferencias) de una población, a partir de una pequeña parte de la misma (muestra).

---
