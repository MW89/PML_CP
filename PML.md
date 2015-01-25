# Activity Quality Prediction
MW  
Friday, January 23, 2015  

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

### Data
There are two data sets: training data (https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and test data (https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).


```r
# Load packages:
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(e1071)
```

```
## Warning: package 'e1071' was built under R version 3.1.2
```

```r
# Load data:
test <- read.csv("pml-testing.csv")
train <- read.csv("pml-training.csv")

# Data dimensions:
dim(test)
```

```
## [1]  20 160
```

```r
dim(train)
```

```
## [1] 19622   160
```

```r
which(names(test)!=names(train))
```

```
## [1] 160
```

```r
names(train)[160]
```

```
## [1] "classe"
```

```r
names(test)[160]
```

```
## [1] "problem_id"
```

```r
# Frequency table of "classe"" variable in the training data set:
table(train$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
After all, there are 160 variables in in both data sets. Only the last variables differs: In the test data set the "classe" variable, which has to be predicted, is actually missing. 


### Goal
The goal was to predict the "classe" variable from the other variables using a machine learning algorithm. 


### Exploratory Data Analysis and Preprocessing

```r
# Which classes were set by read.table?
unique(sapply(train, class))
```

```
## [1] "integer" "factor"  "numeric"
```

```r
# Only look at the relevant belt, forearm, arm and dumbell measurements:
belt.i <- which(grepl("belt", names(test)))
forearm.i <- which(grepl("forearm", names(test)))
arm.i <- which(grepl("arm", names(test)))
dumbell.i <- which(grepl("dumb", names(test)))

# How many variables for each class?
length(belt.i); length(forearm.i); length(arm.i); length(dumbell.i)
```

```
## [1] 38
```

```
## [1] 38
```

```
## [1] 76
```

```
## [1] 38
```

```r
# Indices of all wanted variables:
ok.i <- c(belt.i, forearm.i, arm.i, dumbell.i)

# All variables left over and decision if to include them:
names(train)[-ok.i]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"           "classe"
```

```r
# X = index -> no
# user name: might make sense for this exercicse but not for general context -> no
# raw_timestamp_part_1 & raw_timestamp_part_2 -> no
# cvtd_timestamp -> no
# new_window -> no
# test$num_window -> no

# Remove unwanted columns and set all variables as numeric:
train2 <- sapply(train[,ok.i], as.numeric)
train3 <- cbind(train2, train$classe)

dim(train2)
```

```
## [1] 19622   190
```

```r
dim(train3)
```

```
## [1] 19622   191
```

```r
test3 <- sapply(test[,c(ok.i,160)], as.numeric)

# Test for (near) zero Variances:
nearZeroVar(train3, saveMetric=T)
```

```
##                             freqRatio percentUnique zeroVar   nzv
## roll_belt                    1.101904    6.77810621   FALSE FALSE
## pitch_belt                   1.036082    9.37722964   FALSE FALSE
## yaw_belt                     1.058480    9.97349913   FALSE FALSE
## total_accel_belt             1.063160    0.14779329   FALSE FALSE
## kurtosis_roll_belt        1921.600000    2.02323922   FALSE  TRUE
## kurtosis_picth_belt        600.500000    1.61553358   FALSE  TRUE
## kurtosis_yaw_belt           47.330049    0.01019264   FALSE  TRUE
## skewness_roll_belt        2135.111111    2.01304658   FALSE  TRUE
## skewness_roll_belt.1       600.500000    1.72255631   FALSE  TRUE
## skewness_yaw_belt           47.330049    0.01019264   FALSE  TRUE
## max_roll_belt                1.000000    0.99378249   FALSE FALSE
## max_picth_belt               1.538462    0.11211905   FALSE FALSE
## max_yaw_belt               640.533333    0.34654979   FALSE  TRUE
## min_roll_belt                1.000000    0.93772296   FALSE FALSE
## min_pitch_belt               2.192308    0.08154113   FALSE FALSE
## min_yaw_belt               640.533333    0.34654979   FALSE  TRUE
## amplitude_roll_belt          1.290323    0.75425543   FALSE FALSE
## amplitude_pitch_belt         3.042254    0.06625217   FALSE FALSE
## amplitude_yaw_belt          50.041667    0.02038528   FALSE  TRUE
## var_total_accel_belt         1.426829    0.33126083   FALSE FALSE
## avg_roll_belt                1.066667    0.97339721   FALSE FALSE
## stddev_roll_belt             1.039216    0.35164611   FALSE FALSE
## var_roll_belt                1.615385    0.48924676   FALSE FALSE
## avg_pitch_belt               1.375000    1.09061258   FALSE FALSE
## stddev_pitch_belt            1.161290    0.21914178   FALSE FALSE
## var_pitch_belt               1.307692    0.32106819   FALSE FALSE
## avg_yaw_belt                 1.200000    1.22311691   FALSE FALSE
## stddev_yaw_belt              1.693878    0.29558659   FALSE FALSE
## var_yaw_belt                 1.500000    0.73896647   FALSE FALSE
## gyros_belt_x                 1.058651    0.71348486   FALSE FALSE
## gyros_belt_y                 1.144000    0.35164611   FALSE FALSE
## gyros_belt_z                 1.066214    0.86127816   FALSE FALSE
## accel_belt_x                 1.055412    0.83579655   FALSE FALSE
## accel_belt_y                 1.113725    0.72877383   FALSE FALSE
## accel_belt_z                 1.078767    1.52379982   FALSE FALSE
## magnet_belt_x                1.090141    1.66649679   FALSE FALSE
## magnet_belt_y                1.099688    1.51870350   FALSE FALSE
## magnet_belt_z                1.006369    2.32901845   FALSE FALSE
## roll_forearm                11.589286   11.08959331   FALSE FALSE
## pitch_forearm               65.983051   14.85577413   FALSE FALSE
## yaw_forearm                 15.322835   10.14677403   FALSE FALSE
## kurtosis_roll_forearm      228.761905    1.64101519   FALSE  TRUE
## kurtosis_picth_forearm     226.070588    1.64611151   FALSE  TRUE
## kurtosis_yaw_forearm        47.330049    0.01019264   FALSE  TRUE
## skewness_roll_forearm      231.518072    1.64611151   FALSE  TRUE
## skewness_pitch_forearm     226.070588    1.62572623   FALSE  TRUE
## skewness_yaw_forearm        47.330049    0.01019264   FALSE  TRUE
## max_roll_forearm            27.666667    1.38110284   FALSE  TRUE
## max_picth_forearm            2.964286    0.78992967   FALSE FALSE
## max_yaw_forearm            228.761905    0.22933442   FALSE  TRUE
## min_roll_forearm            27.666667    1.37091020   FALSE  TRUE
## min_pitch_forearm            2.862069    0.87147080   FALSE FALSE
## min_yaw_forearm            228.761905    0.22933442   FALSE  TRUE
## amplitude_roll_forearm      20.750000    1.49322189   FALSE  TRUE
## amplitude_pitch_forearm      3.269231    0.93262664   FALSE FALSE
## amplitude_yaw_forearm       59.677019    0.01528896   FALSE  TRUE
## total_accel_forearm          1.128928    0.35674243   FALSE FALSE
## var_accel_forearm            3.500000    2.03343186   FALSE FALSE
## avg_roll_forearm            27.666667    1.64101519   FALSE  TRUE
## stddev_roll_forearm         87.000000    1.63082255   FALSE  TRUE
## var_roll_forearm            87.000000    1.63082255   FALSE  TRUE
## avg_pitch_forearm           83.000000    1.65120783   FALSE  TRUE
## stddev_pitch_forearm        41.500000    1.64611151   FALSE  TRUE
## var_pitch_forearm           83.000000    1.65120783   FALSE  TRUE
## avg_yaw_forearm             83.000000    1.65120783   FALSE  TRUE
## stddev_yaw_forearm          85.000000    1.64101519   FALSE  TRUE
## var_yaw_forearm             85.000000    1.64101519   FALSE  TRUE
## gyros_forearm_x              1.059273    1.51870350   FALSE FALSE
## gyros_forearm_y              1.036554    3.77637346   FALSE FALSE
## gyros_forearm_z              1.122917    1.56457038   FALSE FALSE
## accel_forearm_x              1.126437    4.04647844   FALSE FALSE
## accel_forearm_y              1.059406    5.11160942   FALSE FALSE
## accel_forearm_z              1.006250    2.95586586   FALSE FALSE
## magnet_forearm_x             1.012346    7.76679238   FALSE FALSE
## magnet_forearm_y             1.246914    9.54031189   FALSE FALSE
## magnet_forearm_z             1.000000    8.57710733   FALSE FALSE
## roll_arm                    52.338462   13.52563449   FALSE FALSE
## pitch_arm                   87.256410   15.73234125   FALSE FALSE
## yaw_arm                     33.029126   14.65701763   FALSE FALSE
## total_accel_arm              1.024526    0.33635715   FALSE FALSE
## var_accel_arm                5.500000    2.01304658   FALSE FALSE
## avg_roll_arm                77.000000    1.68178575   FALSE  TRUE
## stddev_roll_arm             77.000000    1.68178575   FALSE  TRUE
## var_roll_arm                77.000000    1.68178575   FALSE  TRUE
## avg_pitch_arm               77.000000    1.68178575   FALSE  TRUE
## stddev_pitch_arm            77.000000    1.68178575   FALSE  TRUE
## var_pitch_arm               77.000000    1.68178575   FALSE  TRUE
## avg_yaw_arm                 77.000000    1.68178575   FALSE  TRUE
## stddev_yaw_arm              80.000000    1.66649679   FALSE  TRUE
## var_yaw_arm                 80.000000    1.66649679   FALSE  TRUE
## gyros_arm_x                  1.015504    3.27693405   FALSE FALSE
## gyros_arm_y                  1.454369    1.91621649   FALSE FALSE
## gyros_arm_z                  1.110687    1.26388747   FALSE FALSE
## accel_arm_x                  1.017341    3.95984099   FALSE FALSE
## accel_arm_y                  1.140187    2.73672409   FALSE FALSE
## accel_arm_z                  1.128000    4.03628580   FALSE FALSE
## magnet_arm_x                 1.000000    6.82397309   FALSE FALSE
## magnet_arm_y                 1.056818    4.44399144   FALSE FALSE
## magnet_arm_z                 1.036364    6.44684538   FALSE FALSE
## kurtosis_roll_arm          246.358974    1.68178575   FALSE  TRUE
## kurtosis_picth_arm         240.200000    1.67159311   FALSE  TRUE
## kurtosis_yaw_arm          1746.909091    2.01304658   FALSE  TRUE
## skewness_roll_arm          249.558442    1.68688207   FALSE  TRUE
## skewness_pitch_arm         240.200000    1.67159311   FALSE  TRUE
## skewness_yaw_arm          1746.909091    2.01304658   FALSE  TRUE
## max_roll_arm                25.666667    1.47793293   FALSE  TRUE
## max_picth_arm               12.833333    1.34033228   FALSE FALSE
## max_yaw_arm                  1.227273    0.25991234   FALSE FALSE
## min_roll_arm                19.250000    1.41677709   FALSE  TRUE
## min_pitch_arm               19.250000    1.47793293   FALSE  TRUE
## min_yaw_arm                  1.000000    0.19366018   FALSE FALSE
## amplitude_roll_arm          25.666667    1.55947406   FALSE  TRUE
## amplitude_pitch_arm         20.000000    1.49831821   FALSE  TRUE
## amplitude_yaw_arm            1.037037    0.25991234   FALSE FALSE
## roll_forearm.1              11.589286   11.08959331   FALSE FALSE
## pitch_forearm.1             65.983051   14.85577413   FALSE FALSE
## yaw_forearm.1               15.322835   10.14677403   FALSE FALSE
## kurtosis_roll_forearm.1    228.761905    1.64101519   FALSE  TRUE
## kurtosis_picth_forearm.1   226.070588    1.64611151   FALSE  TRUE
## kurtosis_yaw_forearm.1      47.330049    0.01019264   FALSE  TRUE
## skewness_roll_forearm.1    231.518072    1.64611151   FALSE  TRUE
## skewness_pitch_forearm.1   226.070588    1.62572623   FALSE  TRUE
## skewness_yaw_forearm.1      47.330049    0.01019264   FALSE  TRUE
## max_roll_forearm.1          27.666667    1.38110284   FALSE  TRUE
## max_picth_forearm.1          2.964286    0.78992967   FALSE FALSE
## max_yaw_forearm.1          228.761905    0.22933442   FALSE  TRUE
## min_roll_forearm.1          27.666667    1.37091020   FALSE  TRUE
## min_pitch_forearm.1          2.862069    0.87147080   FALSE FALSE
## min_yaw_forearm.1          228.761905    0.22933442   FALSE  TRUE
## amplitude_roll_forearm.1    20.750000    1.49322189   FALSE  TRUE
## amplitude_pitch_forearm.1    3.269231    0.93262664   FALSE FALSE
## amplitude_yaw_forearm.1     59.677019    0.01528896   FALSE  TRUE
## total_accel_forearm.1        1.128928    0.35674243   FALSE FALSE
## var_accel_forearm.1          3.500000    2.03343186   FALSE FALSE
## avg_roll_forearm.1          27.666667    1.64101519   FALSE  TRUE
## stddev_roll_forearm.1       87.000000    1.63082255   FALSE  TRUE
## var_roll_forearm.1          87.000000    1.63082255   FALSE  TRUE
## avg_pitch_forearm.1         83.000000    1.65120783   FALSE  TRUE
## stddev_pitch_forearm.1      41.500000    1.64611151   FALSE  TRUE
## var_pitch_forearm.1         83.000000    1.65120783   FALSE  TRUE
## avg_yaw_forearm.1           83.000000    1.65120783   FALSE  TRUE
## stddev_yaw_forearm.1        85.000000    1.64101519   FALSE  TRUE
## var_yaw_forearm.1           85.000000    1.64101519   FALSE  TRUE
## gyros_forearm_x.1            1.059273    1.51870350   FALSE FALSE
## gyros_forearm_y.1            1.036554    3.77637346   FALSE FALSE
## gyros_forearm_z.1            1.122917    1.56457038   FALSE FALSE
## accel_forearm_x.1            1.126437    4.04647844   FALSE FALSE
## accel_forearm_y.1            1.059406    5.11160942   FALSE FALSE
## accel_forearm_z.1            1.006250    2.95586586   FALSE FALSE
## magnet_forearm_x.1           1.012346    7.76679238   FALSE FALSE
## magnet_forearm_y.1           1.246914    9.54031189   FALSE FALSE
## magnet_forearm_z.1           1.000000    8.57710733   FALSE FALSE
## roll_dumbbell                1.022388   84.20650290   FALSE FALSE
## pitch_dumbbell               2.277372   81.74498012   FALSE FALSE
## yaw_dumbbell                 1.132231   83.48282540   FALSE FALSE
## kurtosis_roll_dumbbell    3843.200000    2.02833554   FALSE  TRUE
## kurtosis_picth_dumbbell   9608.000000    2.04362450   FALSE  TRUE
## kurtosis_yaw_dumbbell       47.330049    0.01019264   FALSE  TRUE
## skewness_roll_dumbbell    4804.000000    2.04362450   FALSE  TRUE
## skewness_pitch_dumbbell   9608.000000    2.04872082   FALSE  TRUE
## skewness_yaw_dumbbell       47.330049    0.01019264   FALSE  TRUE
## max_roll_dumbbell            1.000000    1.72255631   FALSE FALSE
## max_picth_dumbbell           1.333333    1.72765263   FALSE FALSE
## max_yaw_dumbbell           960.800000    0.37203139   FALSE  TRUE
## min_roll_dumbbell            1.000000    1.69197839   FALSE FALSE
## min_pitch_dumbbell           1.666667    1.81429008   FALSE FALSE
## min_yaw_dumbbell           960.800000    0.37203139   FALSE  TRUE
## amplitude_roll_dumbbell      8.000000    1.97227602   FALSE FALSE
## amplitude_pitch_dumbbell     8.000000    1.95189073   FALSE FALSE
## amplitude_yaw_dumbbell      47.920200    0.01528896   FALSE  TRUE
## total_accel_dumbbell         1.072634    0.21914178   FALSE FALSE
## var_accel_dumbbell           6.000000    1.95698706   FALSE FALSE
## avg_roll_dumbbell            1.000000    2.02323922   FALSE FALSE
## stddev_roll_dumbbell        16.000000    1.99266130   FALSE FALSE
## var_roll_dumbbell           16.000000    1.99266130   FALSE FALSE
## avg_pitch_dumbbell           1.000000    2.02323922   FALSE FALSE
## stddev_pitch_dumbbell       16.000000    1.99266130   FALSE FALSE
## var_pitch_dumbbell          16.000000    1.99266130   FALSE FALSE
## avg_yaw_dumbbell             1.000000    2.02323922   FALSE FALSE
## stddev_yaw_dumbbell         16.000000    1.99266130   FALSE FALSE
## var_yaw_dumbbell            16.000000    1.99266130   FALSE FALSE
## gyros_dumbbell_x             1.003268    1.22821323   FALSE FALSE
## gyros_dumbbell_y             1.264957    1.41677709   FALSE FALSE
## gyros_dumbbell_z             1.060100    1.04984201   FALSE FALSE
## accel_dumbbell_x             1.018018    2.16593619   FALSE FALSE
## accel_dumbbell_y             1.053061    2.37488533   FALSE FALSE
## accel_dumbbell_z             1.133333    2.08949139   FALSE FALSE
## magnet_dumbbell_x            1.098266    5.74864948   FALSE FALSE
## magnet_dumbbell_y            1.197740    4.30129447   FALSE FALSE
## magnet_dumbbell_z            1.020833    3.44511263   FALSE FALSE
##                              1.469581    0.02548160   FALSE FALSE
```

```r
# Remove those variables with NZV from data set:
train4 <- as.data.frame(train3[, -nearZeroVar(train3)])
test4 <- as.data.frame(test3[, -nearZeroVar(train3)])

# Check for NAs:
NApercent <- sapply(train4, function(x) sum(is.na(x))/dim(train4)[1]*100)
length(NApercent)
```

```
## [1] 111
```

```r
highNA.i <- which(NApercent > 50)

# Remove according columns with over 50% NAs:
train5 <- train4[,-highNA.i]
test5 <- test4[,-highNA.i]

# Add back the "classe" variable to the training set:
train6 <- data.frame(train5[-66], classe=train$classe)

# Divide training data set for Cross Validation:
set.seed(555)
train.i <- createDataPartition(y = trainPC$classe, p = 0.8, list = FALSE)
trainCV_raw <- train6[train.i,]
testCV_raw <- train6[-train.i,]

rownames(trainCV_raw) <- NULL
rownames(testCV_raw) <- NULL
```

### Model Selection:
I decided to do 10-fold CV (repeated 10 times) and used the trainControl function in R to apply it directly into the train method from the caret package.

```r
CVopt <- trainControl(method="repeatedcv", number=10, repeats=10)
```

I fitted several different models and compared the in sample accuracy to select "the best" model. For Boosting with regressions trees (method="GBM"), linear discriminent analysis (method="LDA"), quadratic discriminent analysis (method="QDA") and decision trees (method="rpart") the in sample accuracy varied between 0.51 (LDA) to 0.83 (rpart). By far the highest accuracy was achieved by the random forest algorithm (ACC=0.99) so i decided to use that model.



```r
modRF <- train(classe ~., preProcess=c("center", "scale"), method="rf",
                data=trainCV_raw, trControl=CVopt)
```




```r
modRF
```

```
## Random Forest 
## 
## 15699 samples
##    65 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (10 fold, repeated 10 times) 
## 
## Summary of sample sizes: 14131, 14128, 14129, 14130, 14128, 14130, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9935981  0.9919016  0.002095539  0.002651144
##   33    0.9931396  0.9913217  0.002113526  0.002673717
##   65    0.9897379  0.9870184  0.002999743  0.003795122
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```


The out of sample accuracy was estimated using the testCV set which was not used for training but includes the "classe" variable as being part of the original training set (see above).


```r
testCV_pred <- predict(modRF, newdata=testCV_raw)
```


```r
confusionMatrix(data=testCV_pred, reference=testCV_raw$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    6    0    0    0
##          B    0  753    8    0    0
##          C    0    0  672    9    0
##          D    0    0    4  634    1
##          E    1    0    0    0  720
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9926         
##                  95% CI : (0.9894, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9906         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9921   0.9825   0.9860   0.9986
## Specificity            0.9979   0.9975   0.9972   0.9985   0.9997
## Pos Pred Value         0.9946   0.9895   0.9868   0.9922   0.9986
## Neg Pred Value         0.9996   0.9981   0.9963   0.9973   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1919   0.1713   0.1616   0.1835
## Detection Prevalence   0.2858   0.1940   0.1736   0.1629   0.1838
## Balanced Accuracy      0.9985   0.9948   0.9898   0.9922   0.9992
```

The out of sample accuracy (estimated with CV) is very high at above 99%. The according error rate therefore is below 1%. The confusion matrix given above includes additional class specific statistics.

### Prediction
I used the same random forest model as above to predict the "classe" variable on the "real" test data set.


```r
test_pred <- predict(modRF, newdata=test5)
```


```r
# Saving test predictions - each in 1 file:
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(test_pred)
```
For this specific test set of 20 observations the automatic control mechanism of the Practical Machine Learning Course at Coursera showed an accuracy of 100% for my model.

### Reference

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3PeCfSJLA



