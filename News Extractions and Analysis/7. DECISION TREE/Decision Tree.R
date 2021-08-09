getwd()
setwd("C://Users//Dwi Nissa//Skripsi//CARI METODE//Preprocessing//FIX//BISMILLAH SUBMIT//5. REMOVE DUPLICATION//OUTPUT")
dat<-read.csv("decision_tree_data.csv")
dat$target<-ifelse(dat$target==0,"No Die","Die")
dat$target<-factor(dat$target)

#Get Factor Data
cart_fac<-c()
cart_fac$day<-factor(dat$day)
cart_fac$time<-factor(dat$time)
cart_fac<-data.frame(cart_fac)
for(i in colnames(dat)[4:length(colnames(dat))]){
  cart_fac[,i] = factor(dat[,i])
}

# Model Performance in 5-Fold CV
library(rpart)
library(rattle)
library(caret)
library(dplyr)

set.seed(123)
train_control<- trainControl(method="cv", number=5, savePredictions = TRUE)
model<- train(target~.-target, data=cart_fac, trControl=train_control, method="rpart")

fold1<-model$pred%>%filter(Resample=='Fold1')
fold2<-model$pred%>%filter(Resample=='Fold2')
fold3<-model$pred%>%filter(Resample=='Fold3')
fold4<-model$pred%>%filter(Resample=='Fold4')
fold5<-model$pred%>%filter(Resample=='Fold5')

c1<-confusionMatrix(fold1$pred, fold1$obs, mode = "prec_recall")
c2<-confusionMatrix(fold2$pred, fold2$obs, mode = "prec_recall")
c3<-confusionMatrix(fold3$pred, fold3$obs, mode = "prec_recall")
c4<-confusionMatrix(fold4$pred, fold4$obs, mode = "prec_recall")
c5<-confusionMatrix(fold5$pred, fold5$obs, mode = "prec_recall")

print(c1)
print(c2)
print(c3)
print(c4)
print(c5)

#Model akhir:
tree<-rpart(target~.-target, data = cart_fac, method="class"
            ,control = list(maxdepth = 4)
)
fancyRpartPlot(tree)