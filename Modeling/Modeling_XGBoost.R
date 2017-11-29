library(xgboost)
library(data.table)



###### Removing tables() ######
rm(list=tables()$NAME)
gc()



###### reading the csv files ######
train_data = fread("traindata_final.csv")
setkey(train_data, msno)

val_data = fread("testdata_final.csv")
setkey(val_data, msno)



###### X & Y ######
x = colnames(train_data[,-c("msno", "is_churn"),with=FALSE])



###### train/test split ######
train_DM <- xgb.DMatrix(data = as.matrix(train_data[,x,with=FALSE]), 
                        label=train_data$is_churn)
valid_DM <- xgb.DMatrix(data = as.matrix(val_data[,x,with=FALSE]), 
                        label=val_data$is_churn)


###### Training ######
seed_used = 1234
param = list(  
  objective           = "binary:logistic", 
  booster             = "gbtree",
  max_depth           = as.integer(length(x)/2),
  eta                 = 0.025,
  gamma               = 0,
  colsample_bytree    = 0.9,
  min_child_weight    = 75,
  subsample           = 0.9,
  seed                = seed_used
)
nrounds = 2000

model = xgb.train(   params              = param, 
                     data                = train_DM,
                     nrounds             = nrounds, 
                     early_stopping_rounds  = 20,
                     watchlist           = list(train=train_DM, val=valid_DM),
                     maximize            = FALSE,
                     eval_metric         = "logloss",
                     print_every_n = 25
)

imp = as.data.frame(xgb.importance(feature_names = x, model = model))
imp

# 0.1725  - 0.236



###### Validation ######
train_pred = predict(model, train_DM)
val_pred = predict(model, valid_DM)

train_pred = data.frame(msno=train_data$msno, is_churn=train_pred)
val_pred = data.frame(msno=val_data$msno, is_churn=val_pred)
head(train_pred)
head(val_pred)

write.csv(train_pred, "xgb_train_pred.csv", row.names = FALSE, quote=FALSE)
write.csv(val_pred,   "xgb_val_pred.csv", row.names = FALSE, quote=FALSE)
