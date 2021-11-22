#Ian Skinner
#2021-11-22
#Objective: predict medical charges per person
#Source: https://www.kaggle.com/mirichoi0218/insurance

#setup===========================================================================================================================================================================
rm(list = ls())
options(scipen = 999)
set.seed(808)

#call pkgs
pacman::p_load(tidyverse, janitor, scales, rio, dplyr, lubridate, gtsummary, caret)
theme_set(theme_classic())

#import data
medical = rio::import("insurance.csv") %>% 
  clean_names() %>% 
  distinct() %>% 
  mutate(id = row_number()) %>% 
  relocate(id)

#data exploration========================================================================================================================================================================
slice_head(medical, n = 10)
summary(medical)

#na evaluatio
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

#recode children
medical = medical %>% 
  mutate(children = as.factor(children))

#dummy varible creation
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
preprocess_vars = c("age", "bmi", "charges")

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

highly_correlated = findCorrelation(cor_mtx,
                                    names = TRUE,
                                    cutoff = 0.7)
