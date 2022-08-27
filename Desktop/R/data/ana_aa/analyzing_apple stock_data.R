# install the packages 
install.packages("modeltime")
library(modeltime)
library(tidymodels)
library(timetk)
library(lubridate)
library(tidyverse)
library(parsnip)
library(rsample)
library(recipes)
library(reshape2)
library(tidyr)
library(ggplot2)
library(plotly)
library(readxl)
library(plyr)
library(lubridate)
library(dplyr)
library(rcartocolor)
library(gghalves)
library(ggdist)
library(workflows)
library(glmnet)
library(randomForest)
library(prophet)
library(quantmod)  

# Download the data for apple  from yahoo finance website 
apple <- getSymbols("AAPL", from="2000-7-1", to='2022-8-1', auto.assign = F) # auto.assign() save it into a data frame 

View(apple)
head(apple)
tail(apple)

# check the names of the columns. Note that the date column is not showing 
colnames(apple)
class(apple)

# we need just only the daily adjusted price and the date. you can see that the date is not part of the 
# variables. So to do that we use the as.data.frame() and create a date variable using the exsiting date in the data
applestock <- as.data.frame(apple) # put the data into a data frame 
applestock$Date <- as.Date(rownames(applestock)) # create a new column and assign the date in the data to it 

#Rename the columns of the downloaded data
colnames(applestock) <- c("Open","High","Low","Close","Volume","Adjusted", "Date") 


# STEP 1: Visualize the data set using the plot_time_series() function and also use the .interactive = TRUE 
# to get an interactive plot. A FALSE will returns a ggplot2 static plot 
applestock %>% 
  plot_time_series(Date, Adjusted, .interactive = TRUE)

# .....................split the data for train and test........................... 

# STEP 2: will use the time_series_split() to split the data into 
splits <- applestock %>%
  time_series_split(assess = "2 year", cumulative = TRUE) # assess = "1 year" tell the 
# the function to use the last 4 months of the data as the testing set. cumulative = TRUE tells the function to the 
# rest of the data as the training set. both the training and testing are contain in the new dataset we created call splits

# Visualize the train and test split 
# using the tk_time_series_cv_plan() convert the split object into a data frame and the 
# plot_time_series_cv_plan() to plot the time series 
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(Date, Adjusted, .interactive = TRUE)

# # STEP 3: Modeling the different models using the train data 
# using the modeltime and parsnip 

# model a:  automatic model

# Auto ARIMA - Auto Arima Model fitting process 
model_fit_arima <-arima_reg() %>% # arima_reg() set up  the general model algorithm and key parameters 
  set_engine("auto_arima") %>% # set_engine() select the specific package-function to use and you can add any function-level arguments here
  fit(Adjusted ~ Date, training(splits)) # the fit() require a data column to be a regressor and the training() comes from the rsample package 

model_fit_arima

# Results 


# Model B: Prophet 
# prophet is like Auto ARIMA, will use the prophet_reg() function 
# and supplying seasonality_yearly = TRUE 
model_fit_prophet <- prophet_reg(seasonality_yearly = TRUE) %>%
  set_engine("prophet") %>%
  fit(Adjusted ~ Date, training(splits))

model_fit_prophet

# results          

# Model C: Machine Learning models 

# Machine learning models are more complex than the automated models. This complexity 
# typically requires a workflow (sometimes called a pipeline in other languages). 
# The general process goes like this:
# a. Create Preprocessing Recipe
# b. Create Model Specifications
# c. Use Workflow to combine Model Spec and Preprocessing, and Fit Model

# Create preprocessing recipe before building the models 
# we first create pre-processing recipe using recipe() and add the time series steps
# The process uses the “date” column to create 45 new features that I’d like to model. 
# These include time-series signature features and fourier series.
recipe_spec <-recipe(Adjusted ~ Date, training(splits)) %>% 
  step_timeseries_signature(Date) %>%
  step_rm(contains("am.pm"), contains("hours"), contains("minute"),
          contains("second"), contains("xts")) %>%
  step_fourier(Date, period = 730, K = 5) %>%
  step_dummy(all_nominal())

recipe_spec %>% prep() %>% juice()

# with the recipe ready, we can now set up our machine learning pipelines 

# Model c1: Elastic Net
# Making an Elastic NET model is easy to do. Just set up your model spec using linear_reg() and set_engine("glmnet").  
# Note that we have not fitted the model yet (as we did in previous steps)

model_spec_glmnet <- linear_reg(penalty = 0.01, mixture = 0.5) %>% 
  set_engine("glmnet")

# Next, make a fitted workflow:
#   Start with a workflow()
# Add a Model Spec: add_model(model_spec_glmnet)
# Add Preprocessing: add_recipe(recipe_spec %>% step_rm(date)) <– Note that I’m removing the “date” 
# column since Machine Learning algorithms don’t typically know how to deal with date or date-time features
# Fit the Workflow: fit(training(splits))

workflow_fit_glmnet <- workflow() %>% 
  add_model(model_spec_glmnet) %>%
  add_recipe(recipe_spec %>% step_rm(Date)) %>%
  fit(training(splits))

# model c2: Random Forest 
# we can fit a Random Forest using a similar process as the Elastic Net
model_spec_rf <- rand_forest(trees = 1000, min_n = 50) %>%
  set_engine("randomForest")

# fit the random forest model with workflow: fit(training(splits))
workflow_fit_rf <- workflow() %>%
  add_model(model_spec_rf) %>%
  add_recipe(recipe_spec %>% step_rm(Date)) %>%
  fit(training(splits))

workflow_fit_rf

# Model c3: Decision Tree 
model_spec_dt <- decision_tree(tree_depth = 10, min_n = 50) %>%
  set_engine("rpart")

# fit the random forest model with workflow: fit(training(splits))
workflow_fit_dt <- workflow() %>%
  add_model(model_spec_dt) %>%
  add_recipe(recipe_spec %>% step_rm(Date)) %>%
  fit(training(splits))

workflow_fit_dt

# Model c4:  Hybrid ML models (prophet boost) 
# this model included several hybrid models (e.g. arima_boost() and prophet_boost() ) 
# that combine both automated algorithms with machine learning. I’ll showcase prophet_boost() next!

# Prophet Boost 
# The Prophet Boost algorithm combines Prophet with XGBoost to get the best of both worlds 
# (i.e. Prophet Automation + Machine Learning). The algorithm works by:
# 1. First modeling the univariate series using Prophet
# 2. Using regressors supplied via the preprocessing recipe (remember our recipe generated 45 new features), 
# and regressing the Prophet Residuals with the XGBoost model
# We can set the model up using a workflow just like with the machine learning algorithms.

model_spec_prophet_boost <- prophet_boost(seasonality_yearly = TRUE) %>%
  set_engine("prophet_xgboost")

# fit with fit(training(splits))
workflow_fit_prophet_boost <- workflow() %>%
  add_model(model_spec_prophet_boost) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))

workflow_fit_prophet_boost

# ........................Model evaluation and selection 

# The Modeltime Workflow
# The modeltime workflow is designed to speed up model evaluation and selection. Now that we 
# have several time series models, let’s analyze them and forecast the future with the modeltime workflow.

# STEP 3: Modeltime Table
# The Modeltime Table organizes the models with IDs and creates generic descriptions to 
# help us keep track of our models. Let’s add the models to a modeltime_table() .
model_table <- modeltime_table(
  model_fit_arima,
  model_fit_prophet,
  workflow_fit_glmnet, 
  workflow_fit_rf,
  workflow_fit_dt,
  workflow_fit_prophet_boost
)

model_table


# STEP 4: Calibration
# Model Calibration is used to quantify error and estimate confidence intervals. 
# We’ll perform model calibration on the out-of-sample data (aka. the Testing Set) with the
# modeltime_calibrate() function. Two new columns are generated (“.type” and “.calibration_data”), 
# the most important of which is the “.calibration_data”. This includes the actual values, fitted values, 
# and residuals for the testing set.
calibration_table <-model_table %>%
  modeltime_calibrate(testing(splits))

calibration_table

# STEP 5:Forecast (Testing Set)
# With calibrated data, we can visualize the testing predictions (forecast).
# Use modeltime_forecast() to generate the forecast data for the testing set as a tibble.
# Use plot_modeltime_forecast() to visualize the results in interactive and static plot formats.
calibration_table %>%
  modeltime_forecast(actual_data = applestock) %>%
  plot_modeltime_forecast(.interactive = TRUE)     

# STEP 6: Accuracy (Testing Set)
# Next, calculate the testing accuracy to compare the models.
# Use modeltime_accuracy() to generate the out-of-sample accuracy metrics as a tibble. 
# Use table_modeltime_accuracy() to generate interactive and static
calibration_table %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = TRUE)


# STEP 7: Analyze Results 
# From the accuracy measures and forecast results, we see that:
#  . Auto ARIMA model is the best model
#  . The worse models are random forest and decision tree 


# STEP 8: Refit and Forecast Forward
# Let’s exclude the Auto ARIMA from our final model, then make future forecasts with the remaining models.
# Refitting is a best-practice before forecasting the future.
# . modeltime_refit() : We re-train on full data ( bike_transactions_tbl )
# . modeltime_forecast() : For models that only depend on the “date” feature, we can use h (horizon) 
# to forecast forward. Setting h = "12 months" forecasts then next 12-months of data.
calibration_table %>%
  # remove ARIM model with low accuracy 
  filter(.model_id != c(4,5)) %>%
  # refit and forecast forward 
  modeltime_refit(applestock) %>%
  modeltime_forecast(h = "12 months", actual_data = applestock) %>%
  plot_modeltime_forecast(.interactive = TRUE)










