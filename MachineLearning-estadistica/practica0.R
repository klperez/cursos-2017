
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(caret)))

# Set seed
set.seed(42)

# Shuffle row indices: rows
rows <- sample(nrow(diamonds))

# Randomly order data
diamonds <- diamonds[rows, ]

# Determine row to split on: split
split <- round(nrow(diamonds) * .80)

# Create train
train <- diamonds[1:split, ]

# Create test
test <- diamonds[(split + 1):nrow(diamonds), ]

# Fit lm model on train: model
model <- lm(price~., data = train)

# Predict on test: p
p <- predict(model, test, type = "response")

# Compute errors: error

error <- p - test$price

# Calculate RMSE

sqrt(mean(error^2))


##################################################################
##################################################################
##################################################################


suppressMessages(suppressWarnings(library(caret)))
suppressMessages(suppressWarnings(library(kernlab)))

data(spam)

inTrain <- createDataPartition(y = spam$type,
                               p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
dim(training)

set.seed(32323)
folds <- createFolds(y = spam$type, k = 10,
                     list = TRUE,returnTrain = TRUE)
sapply(folds,length)

set.seed(32323)
folds <- createFolds(y = spam$type, k = 10,
                     list = TRUE, returnTrain = FALSE)
sapply(folds,length)
folds[[1]][1:10]

set.seed(32323)
folds <- createResample(y = spam$type, times = 10,
                        list = TRUE)
sapply(folds,length)
folds[[1]][1:10]

set.seed(32323)
tme <- 1:1000
folds <- createTimeSlices( y = tme, initialWindow = 20,
                          horizon = 10)
names(folds)
folds$train[[1]]
folds$test[[1]]


set.seed(32343)
modelFit <- train(type ~., data = training, method = "glm")
modelFit

modelFit$finalModel

predictions <- predict(modelFit, newdata = testing)
predictions

confusionMatrix(predictions,testing$type)

#######################################################################
#######################################################################

data(spam)

inTrain <- createDataPartition(y = spam$type,
                               p = 0.75, list = FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]

# Fit lm model using 10-fold CV: model
model <- train(
    type ~ ., training,
    method = "glm",
    trControl = trainControl(
        method = "cv", number = 10,
        verboseIter = TRUE
    )
)

# Print model to console

print(model)


# Fit lm model using 5-fold CV: model
model <- train(
    type ~ ., training,
    method = "glm",
    trControl = trainControl(
        method = "cv", number = 5,
        verboseIter = TRUE
    )
)

# Print model to console
print(model)

# Fit lm model using 5 x 5-fold CV: model
model <- train(
    type ~ ., training,
    method = "glm",
    trControl = trainControl(
        method = "cv", number = 5,
        repeats = 5, verboseIter = TRUE
    )
)

# Print model to console
print(model)

p <- predict(model, testing)

confusionMatrix(p, testing$type)
