
library(gmodels)
library(caret)
library(ROCR)
library(pROC)


loan_status <- factor(sample(x = c(0,1),
                             prob = c(0.35, 0.65), 30000,
                             replace = T))

loan_amount <- round(runif(30000, 500, 35000), 0)

int_rate <- round(runif(30000, 5.42, 23.22), 2)

grade <- factor(sample(LETTERS[1:7], 30000,
                       prob = c(0.33, 0.32, 0.2, 0.11,
                                0.03,0.007, 0.003),
                       replace = T))

emp_length <- round(runif(30000, 0, 62), 0)

home_ownership <- factor(sample(c("MORTAGE", "OTHER", "OWN", "RENT"),
                                30000,
                                prob = c(0.41, 0.01, 0.08, 0.51),
                                replace = T))

annual_inc <- round(runif(30000, 4000, 2039784), 0)

age <- round(runif(30000, 20, 94), 0)


loan_data <- data.frame(loan_status, loan_amount, int_rate, grade,
                    emp_length, home_ownership, annual_inc, age)
## tablas cruzadas

CrossTable(loan_data$loan_status)

CrossTable(x = loan_data$grade,
           y = loan_data$loan_status,
           prop.r = TRUE, prop.c = FALSE,
           prop.t = FALSE, prop.chisq = FALSE)


## Análisis de regresión

inTrain <- createDataPartition(y = loan_data$loan_status, p = 0.7,
                               list = F)

training <- loan_data[inTrain, ]
testing <- loan_data[-inTrain, ]

fitControl <- trainControl( method = "repeatedcv",
                            number = 10,
                            repeats = 10)

modRegLog <- train(loan_status ~.,data = training, method = "glm",
                   trControl = fitControl)

## Predicciones

predRegLog <- predict(modRegLog, newdata = testing)
confusionMatrix(predRegLog, testing$loan_status)

## Aporte de variables

plot(varImp(modRegLog, scale = F), top = 10)

## ROC-Curve

all_probs <- predict(modRegLog, newdata = testing, type = "prob")
probs <- all_probs[,2]
rc <- roc(testing$loan_status, probs)

plot(rc, print.thres = "best",
     print.thres.best.method = "closest.topleft")

# AUC

pred <- predict(modRegLog, newdata = testing)
AUC_model <- auc(testing$loan_status, probs)
