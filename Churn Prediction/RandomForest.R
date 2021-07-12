library(tidyverse)  
library(tidymodels)  
library(skimr)       
library(knitr)       

telco <- read_csv("D:/WA_Fn-UseC_-Telco-Customer-Churn.csv")

telco %>% head() %>% kable()

telco %>% skim()

telco <- telco %>%
  select(-customerID) %>%
  drop_na()

set.seed(seed = 1972) 
train_test_split <-
  rsample::initial_split(
    data = telco,     
    prop = 0.80   
  ) 
train_test_split

train_tbl <- train_test_split %>% training() 
test_tbl  <- train_test_split %>% testing() 

recipe_simple <- function(dataset) {
  recipe(Churn ~ ., data = dataset) %>%
    step_string2factor(all_nominal(), -all_outcomes()) %>%
    prep(data = dataset)
}

recipe_prepped <- recipe_simple(dataset = train_tbl)

train_baked <- bake(recipe_prepped, new_data = train_tbl)
test_baked  <- bake(recipe_prepped, new_data = test_tbl)



#Cross Validation - 10-FoldS

cross_val_tbl <- vfold_cv(train_tbl, v = 10)
cross_val_tbl

cross_val_tbl %>% pluck("splits", 1)

#Random Forest

rf_fun <- function(split, id, try, tree) {
  
  analysis_set <- split %>% analysis()
  analysis_prepped <- analysis_set %>% recipe_simple()
  analysis_baked <- analysis_prepped %>% bake(new_data = analysis_set)
  
  model_rf <-
    rand_forest(
      mode = "classification",
      mtry = try,
      trees = tree
    ) %>%
    set_engine("ranger",
               importance = "impurity"
    ) %>%
    fit(Churn ~ ., data = analysis_baked)
  
  assessment_set     <- split %>% assessment()
  assessment_prepped <- assessment_set %>% recipe_simple()
  assessment_baked   <- assessment_prepped %>% bake(new_data = assessment_set)
  
  tibble(
    "id" = id,
    "truth" = assessment_baked$Churn,
    "prediction" = model_rf %>%
      predict(new_data = assessment_baked) %>%
      unlist()
  )
  
}

pred_rf <- map2_df(
  .x = cross_val_tbl$splits,
  .y = cross_val_tbl$id,
  ~ rf_fun(split = .x, id = .y, try = 3, tree = 200)
)

head(pred_rf)

pred_rf %>%
  conf_mat(truth, prediction) %>%
  summary() %>%
  select(-.estimator) %>%
  filter(.metric %in% c("accuracy", "precision", "recall", "f_meas")) %>%
  kable()

pred_rf %>%
  conf_mat(truth, prediction) %>%
  pluck(1) %>%
  as_tibble() %>%
  
  # Visualize with ggplot
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)

x <- pred_rf %>% conf_mat(truth, prediction)
TN <- x$table[1]/15
FP <- x$table[2]/15
FN <- x$table[3]/15
TP <- x$table[4]/15
churn_rate = TP / FP + TP
churn_rate 

x <- pred_rf %>% conf_mat(truth, prediction)
TN <- x$table[1]/13
FP <- x$table[2]/13
FN <- x$table[3]/13
TP <- x$table[4]/13
churn_rate = TP / FN + TP
churn_rate 



