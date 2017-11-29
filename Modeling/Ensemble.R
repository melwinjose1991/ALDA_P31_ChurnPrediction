library(Metrics)

rf_val_pred = fread("val_rf_pred.csv")
xgb_val_pred = fread("val_xgb_pred.csv")
cat_val_pred = fread("val_cat_pred.csv")

val_data = fread("testdata_final.csv")
val_data = val_data[, c("msno","is_churn"), with=FALSE]



validation = merge(val_data, rf_val_pred, by.x="msno", by.y="msno", all.x=TRUE)
logLoss(validation$is_churn.x, validation$is_churn.y)

validation = merge(val_data, xgb_val_pred, by.x="msno", by.y="msno", all.x=TRUE)
logLoss(validation$is_churn.x, validation$is_churn.y)

validation = merge(val_data, cat_val_pred, by.x="msno", by.y="msno", all.x=TRUE)
logLoss(validation$is_churn.x, validation$is_churn.y)



setnames(rf_val_pred,"is_churn","is_churn_rf")
ensemble = merge(val_data, rf_val_pred, by.x="msno", by.y="msno", all.x=TRUE)
head(ensemble)

setnames(xgb_val_pred,"is_churn","is_churn_xgb")
ensemble = merge(ensemble, xgb_val_pred, by.x="msno", by.y="msno", all.x=TRUE)
head(ensemble)

setnames(cat_val_pred,"is_churn","is_churn_cat")
ensemble = merge(ensemble, cat_val_pred, by.x="msno", by.y="msno", all.x=TRUE)
head(ensemble)



ensemble[,is_churn_ensemble:=(is_churn_rf+is_churn_xgb+is_churn_cat)/3]
logLoss(ensemble$is_churn, ensemble$is_churn_ensemble)
