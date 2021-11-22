#Ian Skinner
#2021-11-22
#Objective: predict medical charges per person

#setup===========================================================================================================================================================================
rm(list = ls())
options(scipen = 999)

#call pkgs
pacman::p_load(tidyverse, janitor, scales, rio, dplyr, lubridate, gtsummary)
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

#variable distributions
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