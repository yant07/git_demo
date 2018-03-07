#used for running on HPC
trainingCate=read.csv("training1.csv")

library(caret)

fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           #summaryFunction = twoClassSummary,
                           savePredictions = T,
                           returnResamp = 'all')

set.seed(825)

cate_rf_1<- train(pdcate2 ~ ., data = trainingCate[,-c(1,3)], 
                  method = "rf", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  metric = "ROC")
Fit=cate_rf_1
save(cate_rf_1, file='NLP_quest_cate_model_rf')



set.seed(825)

cate_rf_1<- train(pdcate2 ~ ., data = trainingCate[,-c(1,3)], 
                  method = "gbm", 
                  trControl = fitControl, 
                  verbose = FALSE, 
                  metric = "ROC")
Fit=cate_rf_1
save(cate_rf_1, file='NLP_quest_cate_model_gbm')

set.seed(825)
thresh_rf_1<- train(as.factor(thresh) ~ ., data = training[,-c(1,2)], 
                    method = "rf", 
                    trControl = fitControl, 
                    verbose = FALSE, 
                    metric = "ROC")
Fit=thresh_rf_1
save(thresh_rf_1, file='NLP_quest_thresh_model_rf')
