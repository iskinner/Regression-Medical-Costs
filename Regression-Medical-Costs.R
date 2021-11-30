#Ian Skinner
#2021-11-22
#Objective: predict medical charges per person, assign insurance rates based on prediction and target profit margin
#Source: https://www.kaggle.com/mirichoi0218/insurance

#setup===========================================================================================================================================================================
rm(list = ls())
options(scipen = 999)
set.seed(808)

#call pkgs
pacman::p_load(tidyverse, janitor, scales, rio, dplyr, lubridate, gtsummary, caret, gbm, modelr)
theme_set(theme_classic())

#import data
medical = rio::import("insurance.csv") %>% 
  clean_names() %>% 
  distinct() %>% 
  mutate(id = row_number()) %>% 
  relocate(id)

#shuffle dataset
medical = slice(medical, sample(1:n()))

#data exploration========================================================================================================================================================================
slice_head(medical, n = 10)
summary(medical)

#note - no one has a charge of exactly zero...
# -> our sample must be of those who went to hospital / had a charge
# -> won't be representative, i.e. those without any medical charges for a period are not represented

#na evaluation
nrow(anti_join(medical, na.omit(medical), by = "id"))

#summary tables
medical %>% 
  select(sex, age, bmi, children, charges) %>% 
  tbl_summary(by = "sex")

medical %>% 
  select(smoker, age, bmi, children, charges) %>% 
  tbl_summary(by = "smoker")

medical %>% 
  select(region, age, bmi, children, charges) %>% 
  tbl_summary(by = "region")

#continuous variable distributions
dist_fn = function(df, varname) {
  ggplot(data = df,
         aes(x = {{varname}})) +
    geom_histogram() +
    labs(title = "Variable distribution") +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma)
}

dist_fn(medical, age)
dist_fn(medical, bmi)
dist_fn(medical, charges)

#discrete and categorical variables
disc_fn = function(df, varname) {
  df %>% 
    count({{varname}}) %>% 
    mutate(pct = n / sum(n)) %>% 
    ggplot(aes(x = {{varname}},
               y = pct)) +
    geom_col(width = .5) +
    scale_y_continuous(labels = percent)
}

disc_fn(medical, sex)
disc_fn(medical, smoker)
disc_fn(medical, region)

#recode children, take natural log of medical charges to create normal distribution
medical = medical %>% 
  mutate(children = as.factor(children),
         charges = log(charges))

medical %>% 
  ggplot(aes(x = charges)) +
  geom_histogram()

#data preprocessing============================================================================================================================================================
#dummy variable creation
dummy_vars = c("sex", "smoker", "region", "children")

dummy_df = medical %>% select(all_of(dummy_vars))
medical_df = medical %>% select(!all_of(dummy_vars))

dummy_df = data.frame(predict(dummyVars(data = dummy_df, 
                                        "~.", 
                                        sep = "_"), 
                              dummy_df))

#join data back together
medical = bind_cols(medical_df, dummy_df)

#variables for preprocessing
preprocess_vars = c("age", "bmi")

preprocess_df = medical %>% select(all_of(preprocess_vars))
medical_df = medical %>% select(!all_of(preprocess_vars))

preprocess_df = data.frame(predict(preProcess(x = preprocess_df,
                                              method = c("center", "scale")),
                                   preprocess_df))

#join data back together
medical = bind_cols(medical_df, preprocess_df) %>% 
  select(!id)

#correlations
cor_mtx = cor(medical)

findCorrelation(cor_mtx, names = TRUE, cutoff = 0.7)

#remove highly correlated, duplicative, or unnecessary variables
medical = medical %>% 
  select(!c(sexmale, smokerno))

#find near zero variance variables
nearZeroVar(medical, names = TRUE)

#fix near zero variance columns by re-assigning to new column with a bigger bucket
medical = medical %>% 
  mutate(children_3_over = children_3 + children_4 + children_5) %>%
  select(!c(children_3, children_4, children_5)) %>% 
  relocate(children_3_over, .after = "children_2")

#check if we got rid of nzv issue
nearZeroVar(medical, names = TRUE)

#check if we made any new large correlations
cor_mtx = cor(medical)
findCorrelation(cor_mtx, names = TRUE, cutoff = 0.7)

#rename some variables
medical = medical %>% 
  rename(is_smoker = smokeryes,
         is_female = sexfemale,
         region_northeast = regionnortheast,
         region_northwest = regionnorthwest,
         region_southeast = regionsoutheast,
         region_southwest = regionsouthwest,
         no_children = children_0,
         one_child = children_1,
         two_children = children_2,
         more_than_two_kids = children_3_over)

#machine learning================================================================================================================================================

#split data into training, testing and validation sets
split = createDataPartition(medical$charges, p = 0.7, list = F)

#store validation set
train = data.frame(medical[split,])
medical = data.frame(medical[-split,])

#split again for training and test set
split = createDataPartition(medical$charges, p = 0.67, list = F)

#store train and test set
test = data.frame(medical[split,])
validation = data.frame(medical[-split,])

#check distributions of outcome variable to make sure our datasets are balanced
lattice::histogram(train$charges)
lattice::histogram(test$charges)
lattice::histogram(validation$charges)

#initialize empty data frame
results = data.frame(method = as.character(),
                     name = as.character(),
                     optimized = as.character(),
                     train_rmse = as.numeric(),
                     train_mae = as.numeric(),
                     test_rmse = as.numeric(),
                     test_mae = as.numeric(),
                     test_mape = as.numeric())

#linear regression - unoptimized====================================================================================================================================
method = "lm"
optimized = "N"
name = "Linear Regression"

#train model
model = train(data = train,
              charges ~ .,
              method = method)

#check out model
model

#in sample results
train_rmse = model$results$RMSE
train_mae = model$results$MAE

ggplot(varImp(model)) +
  labs(title = paste0("Feature importance plot, model type: ", name),
       subtitle = "Importance scaled to 100",
       y = "Feature importance")

test_rmse = rmse(model, test)
test_mae = mae(model, test)
test_mape = mape(model, test)

model_results = data.frame(method, name, optimized, train_rmse, train_mae, test_rmse, test_mae, test_mape)

results = bind_rows(results, model_results)

#save version of model for validation set later
lm_model = model

#gradient boosting - unoptimized=======================================================================================================================================
method = "gbm"
optimized = "N"
name = "Gradient Boosting"

#train model
model = train(data = train,
              charges ~ .,
              method = method)

#check out model
model

#in sample results
train_best = row.names(model$bestTune)
train_rmse = model$results[train_best,]$RMSE
train_mae = model$results[train_best,]$MAE

ggplot(varImp(model)) +
  labs(title = paste0("Feature importance plot, model type: ", name),
       subtitle = "Importance scaled to 100",
       y = "Feature importance")

test_rmse = rmse(model, test)
test_mae = mae(model, test)
test_mape = mape(model, test)

model_results = data.frame(method, name, optimized, train_rmse, train_mae, test_rmse, test_mae, test_mape)

results = bind_rows(results, model_results)

#save version of model for validation set later
gbm_model_v1 = model

#gradient boosting - optimized=======================================================================================================================================
method = "gbm"
optimized = "Y"
name = "Gradient Boosting"

#set up 10-fold cross valiation
trControl = trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 10,
                         savePredictions = TRUE)

#custom tuning parameters for gradient boosting model
tuneGrid = expand.grid(interaction.depth = c(1, 3, 5, 7),
                       n.trees = (1:10) * 10,
                       shrinkage = 0.05,
                       n.minobsinnode = 5)

#train model
model = train(data = train,
              charges ~ .,
              method = method,
              tuneGrid = tuneGrid,
              # tuneLength = 1,
              trControl = trControl)

#check out model
model
ggplot(model) +
  labs(title = "Modeling error results over n iterations: gradient boosting method")

#in sample results
train_best = row.names(model$bestTune)
train_rmse = model$results[train_best,]$RMSE
train_mae = model$results[train_best,]$MAE

ggplot(varImp(model)) +
  labs(title = paste0("Feature importance plot, model type: ", name),
       subtitle = "Importance scaled to 100",
       y = "Feature importance")

test_rmse = rmse(model, test)
test_mae = mae(model, test)
test_mape = mape(model, test)

model_results = data.frame(method, name, optimized, train_rmse, train_mae, test_rmse, test_mae, test_mape)

results = bind_rows(results, model_results)

#save version of model for validation set later
gbm_model_v2 = model

results

#validation==================================================================================================================================================================

#try out models on validation set
rmse_v_lm = rmse(lm_model, validation)
rmse_v_gbmv1 = rmse(gbm_model_v1, validation)
rmse_v_gbmv2 = rmse(gbm_model_v2, validation)

rmse_v_lm
rmse_v_gbmv1
rmse_v_gbmv2

results

#implementation=====================================================================================================================================================================
#let's assign some insurance rates based on the predicted costs for our customers based on the best model
#pretend we don't know the insurance rates for our validation set

new_customers = validation %>% 
  select(!c(charges))

#get optimal trees from best tune
optimal_trees = as.numeric(gbm_model_v2$bestTune[1])

#make 'new' predictions
optimal_predictions = data.frame(predict(object = gbm_model_v2, 
                                         newdata = new_customers, 
                                         n.trees = optimal_trees)) %>% 
  rename(predicted_charges = 1)

#bind predictions to new customers, convert predictions to dollar value
new_customers = bind_cols(new_customers, optimal_predictions) %>% 
  mutate(predicted_charges = exp(predicted_charges))

slice_head(new_customers, n = 25)
