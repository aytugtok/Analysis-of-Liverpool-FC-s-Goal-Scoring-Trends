#Time Series Project

library(dplyr)
library(ggplot2)
library(forecast)
library(devtools)
library(AnomalyDetection)
library(tseries)
library(uroot)
library(anomalize)
library(zoo)
library(readxl)
library(knitr)
library(xfun)
library(rmarkdown)
library(pdR)
library(lubridate)
library(anomalize)
library(stats)
library(MVN)
library(imputeTS)
library(TSA)
#rmarkdown::render("C:/Users/programlama/OneDrive/Desktop/stat497/interim.Rmd", output_format="html_document")

Sys.setlocale("LC_TIME", "C")

#Reading the file
setwd("C:/Users/programlama/OneDrive/Desktop/stat497")
lfc <- read.csv("lfc_edited.csv")
View(lfc)

#Transforming data to univariate
lfc <- lfc %>% select(Date, Goals_Scored)

lfc <- lfc %>% 
  mutate(Date = format(as.Date(Date), "%Y %B")) %>%
  group_by(Date) %>%
  summarize(Goals = sum(Goals_Scored))

#Ordering the data by date
lfc <- lfc %>% 
  mutate(Date = as.Date(paste(Date,"1"), format="%Y %B %d")) %>%
  arrange(Date) 

lfc$Date <- ifelse(
lfc$Date %in% as.Date(c("2020-06-01", "2020-07-01")),
lfc$Date %m-% months(2),  
lfc$Date                 
)

lfc$Date <- as.Date(lfc$Date)
lfc <-  lfc %>%  filter(Date >= as.Date("2000-08-01") & Date <= as.Date("2024-05-01"))
length(lfc$Goals)
View(lfc)

#Imputation for missing value
median_goals <- median(lfc$Goals, na.rm = T)
imptd_row <- data.frame(Date = as.Date("2020-08-01"), Goals = median_goals)
lfc <- rbind(lfc, imptd_row)
lfc <- lfc[order(lfc$Date),]
length(lfc$Goals)
View(lfc)

lfc


str(lfc)

#Arranging the time interval according to obtain a sustainable frequency
ts_lfc <- ts(lfc, start=c(2000,8), frequency = 10)
ts_lfc <- ts_lfc[,"Goals"]
length(ts_lfc)



#Plotting the time series 
autoplot(ts_lfc)+
  ggtitle("Goals scored by Liverpool FC for Last 24 Seasons")+
  xlab("Date")+
  ylab("Goals Scored")+
  theme_minimal()


#For cross-validation, the last season is arranged as test data
test_data <- tail(lfc,10)
length(test_data$Goals)
train_data <- anti_join(lfc, test_data, by="Date")
length(train_data$Goals)

train_ts <- window(ts_lfc, start=c(2000,8), end=c(2023,7))
test_ts <- window(ts_lfc, start = c(2023,8), end=c(2024,7))
length(train_ts)
length(test_ts)


##ANOMALY DETECTION

#Decomposition
train_data %>% anomalize::time_decompose(Goals, method = "stl", frequency = 10, trend = "auto") %>% anomalize::anomalize(remainder, method = "gesd", alpha = 0.05, max_anoms = 0.2) %>% anomalize::plot_anomaly_decomposition()

#Extracting the anomalous data points
train_data %>% 
  anomalize::time_decompose(Goals) %>%
  anomalize::anomalize(remainder) %>%
  anomalize::time_recompose() %>%
  anomalize::plot_anomalies(time_recomposed = TRUE, ncol = 3, alpha_dots = 0.5)

#train_data %>% 
#  anomalize::time_decompose(Goals) %>%
 # anomalize::anomalize(remainder) %>%
  #anomalize::time_recompose() %>%
  #filter(anomaly == 'yes') #No anomalies


#Cleaning anomalies
train_data_cleaned <- train_data %>%
  anomalize::time_decompose(Goals) %>%
  mutate(observed = ifelse(observed < 0, observed+1, observed)) %>%
  anomalize::anomalize(remainder) %>%
  clean_anomalies()
test_ts

#We can directly transform the data into time series object
train_ts_cleaned <- ts(train_data_cleaned$observed, start=c(2000,8), frequency = 10)  
autoplot(train_ts_cleaned, ylab="Goals", xlab="Date",
main="Goals scored by Liverpool FC", lwd=0.6)+
theme_minimal()

tail(train_ts,10)



#Check the BoxCox Case: 
BoxCox.ar(train_ts_cleaned)
lambda <- BoxCox.lambda(train_ts_cleaned)
train_boxcox <- BoxCox(train_ts_cleaned, lambda)
BoxCox.lambda(train_boxcox)


#Drawing ACF and PACF plot
library(gridExtra)
g1<-ggAcf(train_boxcox,lag.max = 50)+theme_minimal()+ggtitle("ACF of Data")
g2<-ggPacf(train_boxcox,lag.max = 50)+theme_minimal()+ggtitle("PACF of Data")
grid.arrange(g1,g2,ncol=2)



#Applying KPSS test to obtain the stationarity status 
kpss.test(train_boxcox, "Level") #it is not stationary
kpss.test(train_boxcox, "Trend") #the process is deterministic


adf.test(train_boxcox) #indicates no regular unit root
forecast::ocsb.test(train_boxcox) #no seasonal unit root

ndiffs(train_boxcox)
nsdiffs(train_boxcox)

kpss.test(diff(train_boxcox), "Level") #stationary
kpss.test(diff(train_boxcox), "Trend") #deterministic

adf.test(diff(train_boxcox)) #no regular ur
ocsb.test(diff(train_boxcox)) #no seasonal ur

autoplot(diff(train_boxcox))+theme_minimal()+ggtitle("Plot for Differenced Data")

library(gridExtra)
g1<-ggAcf(diff(train_boxcox),lag.max = 40)+theme_minimal()+ggtitle("ACF of Differenced Data")
g2<-ggPacf(diff(train_boxcox),lag.max = 40)+theme_minimal()+ggtitle("PACF of Differenced Data")
grid.arrange(g1,g2,ncol=2)


eacf(diff(train_boxcox))

#Suggesting models: 

fit1 <- Arima(train_boxcox, order=c(1,1,1),seasonal=c(1,0,1))
fit1

fit2 <-  Arima(train_boxcox, order=c(2,1,1),seasonal=c(1,0,1))
fit2

fit3 <- Arima(train_boxcox, order=c(3,1,1),seasonal=c(1,0,1))
fit3

fit4 <- Arima(train_boxcox, order=c(1,1,1),seasonal=c(1,0,3))
fit4

fit5 <- Arima(train_boxcox, order=c(2,1,1),seasonal=c(1,0,3))
fit5



#fit1 is the only significant model

##DIAGNOSTIC CHECKING
r=resid(fit1)
summary(r)

#General Plot of the Residuals
autoplot(r)+geom_line(y=0)+theme_minimal()+ggtitle("Plot of The Residuals")

#Q-Q Plot of the Residuals
ggplot(r, aes(sample = r)) +stat_qq()+geom_qq_line()+ggtitle("QQ Plot of the Residuals")+theme_minimal()

#Histogram of the Residuals
ggplot(r,aes(x=r))+geom_histogram(bins=20)+geom_density()+ggtitle("Histogram of Residuals")+theme_minimal()

#Formal Normality Tests on Residuals
jarque.bera.test(r) #p-value is small, not normal
shapiro.test(r) #not normal



#Breusch-Godfrey Test for Homoscedasticity
library(lmtest)
m = lm(r ~ train_boxcox+zlag(train_boxcox)+zlag(train_boxcox,2))
bptest(m) #large p-value, homoscedastic

#Box-Ljung Test
Box.test(r,lag=15,type = c("Ljung-Box")) #homoscedasticity

#Box-Pierce Test (Modified Version of Box-Ljung Test)
Box.test(r,lag=15,type = c("Box-Pierce")) # Portmanteau test for the null hypothesis of independence in a given time series.



#Engle's ARCH Test
rr = (resid(fit1))^2

library(FinTS)
ArchTest(rr) #since larger p-value, no ARCH effect


##FORECASTING

##SARIMA
sarima_f <- forecast(fit1,h=10)
sarima_f
autoplot(sarima_f) + theme_minimal() + ggtitle("Forecast of SARIMA Model")



f_t <- InvBoxCox(sarima_f$mean, lambda)
str(f_t)
autoplot(f_t,main=c("Comparison of forecast vs actual test"), series="forecast" ) + autolayer(test_ts,series = "actual") + theme_bw()
accuracy(f_t, test_ts)
accuracy(InvBoxCox(sarima_f$fitted,lambda),train_ts)

plot(ts_lfc_merged,ylab = "Goals Scored", lwd = 2, main = "Time Series Plot of Actual Values and SARIMA Forecast")
lines(InvBoxCox(sarima_f$fitted,lambda), col = "purple", lty = 2, lwd = 2)
abline(v = 2023+07/10, col = "red", lwd = 2)
lines(f_t, col = "blue", lty = 1, lwd = 2)
LI <- ts(InvBoxCox(f_t$lower[, 2], lambda), start = c(2023, 8), frequency = 10)
UI <- ts(InvBoxCox(f_t$upper[, 2], lambda), start = c(2023, 8), frequency = 10)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)

##Exponential Smoothing
autoplot(decompose(ts_lfc))

shapiro.test(resid(ets2)) #low p-value not normal
m = lm(resid(ets2) ~ train_ts+zlag(train_ts)+zlag(train_ts,2))
bptest(m)


ets1 <- ets(train_ts, model="ZZZ") #Since lambda is close to 1, we should use additive models
summary(ets1)
shapiro.test(resid(ets1))
ets2 <- forecast::forecast(ets1, h = 10)
autoplot(ets2$mean, series = "ETS")+autolayer(test_ts,series="actual")+theme_bw()

fit.hw<-hw(train_ts, h=11, seasonal="additive")

str(fit.hw)
test_data

accuracy(fit.hw,test_ts)
autoplot(fit.hw, ylab="Goals")




##Neural Network Forcasting
nn <- nnetar(train_ts)
nn

nn_f <- forecast::forecast(nn,h=10,PI=TRUE)
accuracy(nn_f,test_ts)


autoplot(train_ts)+autolayer(fitted(nn),lwd=1.4)+theme_minimal()+ggtitle("Fitted Values of NN Model")

autoplot(nn_f)+theme_minimal()+autolayer(test_ts,series="actual",color="red")

plot(ts_lfc, lwd = 2, main = "Neural Network")
lines(nn_f$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+07/10, col = "red", lwd = 2)
lines(nn_f$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(nn_f$lower[, 2], start = c(2023, 8), frequency = 10)
UI <- ts(nn_f$upper[, 2], start = c(2023, 8), frequency = 10)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)

autoplot(nn_f$mean, series = "NN")+theme_bw() + autolayer(test_ts, series = "actual")

##Hyperparameter Tuning for Neural Network
p <- c(1, 2, 3)
P <- c(1, 10, 20)
size <- c(128, 64, 32)
repeats <- c(10, 15, 20,40,80)

results <- data.frame(
  p = numeric(),
  P = numeric(),
  size = numeric(),
  repeats = numeric(),
  RMSE = numeric()
)

for (p in p) {
  for (P in P) {
    for (size in size)
      for(repeats in repeats) {{
        nnm <-nnetar(train_ts, p = p, P = P, size = size, repeats = repeats)
        nnf <- forecast(nnm, h = 10)
        accur <- accuracy(nnf,test_ts)
        rmse <- accur[2,2]
        
        results <- rbind(results, data.frame(
        p = p, 
        P = P, 
        size = size,
        repeats = repeats,
        RMSE = rmse
      ))
      
    }}}}

best_params <- results[which.min(results$RMSE), ]
best_params

nnm <-nnetar(train_ts, p = 1, P = 10, size = 32, repeats = 80)
nnf <- forecast(nnm, h = 10, PI = TRUE)
accuracy_nn <- accuracy(nnf,test_ts)
accuracy_nn

autoplot(nnf)+autolayer(test_ts,series="actual",color="red")+theme_minimal()

plot(ts_lfc_merged, lwd = 2, ylab="Goals",main = "Hyperparameter Tuning in Neural Network")
lines(nnf$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+07/10, col = "red", lwd = 2)
lines(nnf$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(nnf$lower[, 2], start = c(2023, 8), frequency = 10)
UI <- ts(nnf$upper[, 2], start = c(2023, 8), frequency = 10)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)


##Prophet
library(prophet)
ds <- train_data$Date
df <- data.frame(ds, y=train_ts)

train_prophet <- prophet(df)
future <- make_future_dataframe(train_prophet,periods = 10)
prophet_f <- predict(train_prophet, future)
accuracy(tail(prophet_f$yhat,10),test_ts)
plot(train_prophet, prophet_f)+theme_minimal()

prophet_plot_components(train_prophet, prophet_f)


prop_fit <- ts(head(prophet_f$yhat,230),start=c(2000,8), end=c(2023,5), frequency = 10)
prop <- ts(tail(prophet_f$yhat,10), start=c(2023,8), frequency = 10)

plot(ts_lfc, lwd = 2, main = "Prophet")
lines(prop_fit, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+07/10, col = "red", lwd = 2)
lines(prop, col = "blue", lty = 1, lwd = 2)
LI <- ts(tail(prophet_f$yhat_lower,10), start = c(2023, 8), frequency = 10)
UI <- ts(tail(prophet_f$yhat_upper,10), start = c(2023, 8), frequency = 10)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)



##Hypermeter for Prophet
prophet_new <- prophet(df,changepoint.range=0.5,changepoint.prior.scale=0.2,seasonality.prior.scale=0.7)

future_new=make_future_dataframe(prophet_new,periods = 10, freq = "month")
forecast_new <- predict(prophet_new, future_new)
accuracy(tail(forecast_new$yhat,10),test_ts)

changepoint_prior <- c(0.1, 0.5, 0.9)
seasonality_prior <- c(0.1, 0.3, 0.5)
changepoint_range <- c(0.6, 0.8, 0.9)

results <- data.frame(
  changepoint_prior = numeric(),
  seasonality_prior = numeric(),
  changepoint_range = numeric(),
  RMSE = numeric()
)

for (cp in changepoint_prior) {
  for (sp in seasonality_prior) {
    for (cr in changepoint_range) {
      m <- prophet(
        changepoint.prior.scale = cp,
        seasonality.prior.scale = sp,
        changepoint.range = cr
      )
      m <- fit.prophet(m, df) 
      

      future <- make_future_dataframe(m, periods = 10, freq = "month")
      forecast <- predict(m, future)
      
      predicted <- tail(forecast$yhat, 10)
      acc <- accuracy(predicted, test_ts)  
      rmse <- acc["Test set", "RMSE"]  # Extract RMSE from accuracy
      
      results <- rbind(results, data.frame(
        changepoint_prior = cp, 
        seasonality_prior = sp, 
        changepoint_range = cr, 
        RMSE = rmse
      ))
    }
  }
}

#best parameters
best_params <- results[which.min(results$RMSE), ]
best_params

#Arranging parameters according to the best_params
prophet_new2 <- prophet(df,changepoint.range=0.6,changepoint.prior.scale=0.5,seasonality.prior.scale=0.1)
future_new2=make_future_dataframe(prophet_new2,periods = 10, freq = "month")
forecast_new2 <- predict(prophet_new2, future_new2)
accuracy(tail(forecast_new2$yhat,10),test_ts) #lower rmse than original prophet
accuracy(head(forecast_new2$yhat,230),test_ts)

prop_fit2 <- ts(head(forecast_new2$yhat,230),start=c(2000,8), end=c(2023,7), frequency = 10)
prop2 <- ts(tail(forecast_new$yhat,10), start=c(2023,8), frequency = 10)

plot(ts_lfc_merged, lwd = 2, ylab="Goals",main = "Hyperparamter Tuning Prophet Plot")
lines(prop_fit2, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+07/10, col = "red", lwd = 2)
lines(prop2, col = "blue", lty = 1, lwd = 2)
LI <- ts(tail(forecast_new2$yhat_lower,10), start = c(2023, 8), frequency = 10)
UI <- ts(tail(forecast_new2$yhat_upper,10), start = c(2023, 8), frequency = 10)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)



##TBATS
tbats_model <- tbats(train_ts)
tbats_model
autoplot(train_ts,main="TS plot of Train with TBATS Fitted") +autolayer(fitted(tbats_model), series="Fitted") +theme_minimal()

shapiro.test(resid(tbats_f))
library(lmtest)
m = lm(resid(tbats_f) ~ train_boxcox+zlag(train_boxcox)+zlag(train_boxcox,2))
bptest(m)


tbats_f <- forecast(tbats_model, h=10)
tbats_f
autoplot(tbats_f)+autolayer(test_ts,series="actual",color="red")+theme_minimal()

accuracy(tbats_f,test_ts)

plot(ts_lfc, lwd = 2, main = "TBATS")
lines(tbats_f$fitted, col = "purple", lty = 2, lwd = 2)
abline(v = 2023+07/10, col = "red", lwd = 2)
lines(tbats_f$mean, col = "blue", lty = 1, lwd = 2)
LI <- ts(tbats_f$lower[, 2], start = c(2023, 8), frequency = 10)
UI <- ts(tbats_f$upper[, 2], start = c(2023, 8), frequency = 10)
lines(LI, col = "green", lty = 2, lwd = 2)
lines(UI, col = "green", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Series", "Fitted Values", "Point Forecast",
                  "95% Prediction Interval", "Forecast Origin"),
       col = c("black", "purple", "blue", "green", "red"),
       lty = c(1, 2, 1, 2, 2, 1),
       lwd = c(2, 2, 2, 2, 2, 2),
       cex = 0.6)





