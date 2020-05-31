setwd("D:/AI & Data Science/R/Project 2 Retail")

store_test=read.csv("store_test.csv",stringsAsFactors = F)
store_train=read.csv("store_train.csv", stringsAsFactors = F)

library(dplyr)
library(car)

store_test$store=NA

store_test$data="test"
store_train$data="train"

store_all=rbind(store_train,store_test)

lapply(store_all,function(x) sum(is.na(x)))
n=table(store_train$state_alpha)
sort(n)

for(col in names(store_all)){
  if(sum(is.na(store_all[,col]))>0 & (col %in% c("population","country"))){
    store_all[is.na(store_all[,col]),col]=round(mean(store_all[,col],na.rm=T))
  }
}

glimpse(store_all)

n=table(store_all$store_Type)
dim(n)
n

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

store_all=CreateDummies(store_all ,"store_Type",100)
glimpse(store_all)

cat_var=names(store_all)[sapply(store_all,is.character)]
cat_var
store_all=store_all %>% 
  select(-Id,-countyname,-storecode,-Areaname,-countytownname,-state_alpha)
store_train=store_all %>% filter(data=='train') %>% select(-data)
store_test=store_all %>% filter(data=='test') %>% select(-data,-store)

glimpse(store_train)

set.seed(2)
s=sample(1:nrow(store_train),.75*nrow(store_train))
store_train1=store_train[s,]
store_train2=store_train[-s,]

library(pROC)

#Logistic Regression Model
for_vif=lm(store~.-Id-countyname-storecode-Areaname
           -countytownname-state_alpha,data=store_train1)

sort(vif(for_vif),decreasing = T)

for_vif=lm(store~.-Id-countyname-storecode-Areaname
           -countytownname-state_alpha-sales0-sales2-sales3-sales1,data=store_train1)

log_fit=glm(store~.-Id-countyname-storecode-Areaname
            -countytownname-state_alpha,data=store_train1)

log_fit
log_fit=step(log_fit)

summary(log_fit)

formula(log_fit)

val.score=predict(log_fit,newdata = store_train2,type='response')

auc(roc(store_train2$store,val.score))

write.table(val.score,file ="Abhilash_Singh_P2_part2.csv",
            row.names = F,col.names="store")

#DTree Model
library(rpart)
library(rpart.plot)
library(tidyr)
library(randomForest)
require(rpart)
library(tree)

dtModel = tree(store~.-Id-countyname-storecode-Areaname
               -countytownname-state_alpha,data=store_train1)
plot(dtModel)
dtModel

val.score=predict(dtModel,newdata = store_train2)
auc(roc(store_train2$store,val.score))
class(store_train$store)
#Random Forest Model
?randomForest
randomForestModel = randomForest(store~.,data=store_train1)
d=importance(randomForestModel)
d
names(d)
d=as.data.frame(d)
d$IncNodePurity=rownames(d)
d %>% arrange(desc(IncNodePurity))

val.score=predict(randomForestModel,newdata = store_train2)
auc(roc(store_train2$store,val.score))

write.table(val.score,file ="Abhilash_Singh_P2_part2.csv",
            row.names = F,col.names="store")

#GBM Model
library(gbm)
library(cvTools)

param=list(interaction.depth=c(1:10),
           n.trees=c(50,100,200,500,700),
           shrinkage=c(.1,.01,.001),
           n.minobsinnode=c(1,2,5,10))

subset_paras=function(full_list_para,n=10){
  
  all_comb=expand.grid(full_list_para)
  
  s=sample(1:nrow(all_comb),n)
  
  subset_para=all_comb[s,]
  
  return(subset_para)
}

num_trials=50
my_params=subset_paras(param,num_trials)

mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(score)
}

myauc=0

for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  
  params=my_params[i,]
  
  k=cvTuning(gbm,store~.,
             data =store_train,
             tuning =params,
             args = list(distribution="bernoulli"),
             folds = cvFolds(nrow(store_train), K=10, type = "random"),
             cost =mycost_auc, seed =2,
             predictArgs = list(type="response",n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this>myauc){
    print(params)
    
    myauc=score.this
    print(myauc)
    # uncomment the line above to keep track of progress
    best_params=params
  }
  
  print('DONE')
  # uncomment the line above to keep track of progress
}

myauc

best_params

best_params=data.frame(interaction.depth=10,
                       n.trees=700,
                       shrinkage=0.01,
                       n.minobsnode=2)

store.gbm.final=gbm(store~.,data=store_train,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "bernoulli")

store.gbm.final

test.pred=predict(store.gbm.final,newdata=store_test,
                  n.trees = best_params$n.trees,type="response")

write.table(test.pred,file ="Abhilash_Singh_P2_part2_GBM.csv",
            row.names = F,col.names="store")


##########Final Submission#############
for(i in 1:num_trials){
  print(paste0('starting iteration:',i))
  params=my_params[i,]
  k=cvTuning(gbm,store~.-Suburb-Address-SellerG-CouncilArea,
             data =store_train,
             tuning =params,
             args = list(distribution="bernoulli"),
             folds = cvFolds(nrow(store_train), K=10, type = "random"),
             seed =2,
             predictArgs = list(n.trees=params$n.trees)
  )
  score.this=k$cv[,2]
  
  if(score.this<myerror){
    print(params)
    myerror=score.this
    print(myerror)
    best_params=params
  }
  
  print('DONE')
}

myerror
best_params

best_params=data.frame(interaction.depth=7,
                       n.trees=700,
                       shrinkage=0.1,
                       n.minobsnode=2)

bs.gbm.final=gbm(store~.-Id-countyname-storecode-Areaname
                 -countytownname-state_alpha,data=store_train,
                 n.trees = best_params$n.trees,
                 n.minobsinnode = best_params$n.minobsnode,
                 shrinkage = best_params$shrinkage,
                 interaction.depth = best_params$interaction.depth,
                 distribution = "bernoulli")
bs.gbm.final

test.pred=predict(bs.gbm.final,newdata=store_test,
                  n.trees = best_params$n.trees)

write.table(test.pred,file ="Abhilash_Singh_P2_part2.csv",
            row.names = F,col.names="store")


############################Quiz############################################
#1
sum(store_train %>% 
  mutate(total_sales=sales0+sales1+sales2+sales3+sales4) %>% 
  filter(Areaname=="Kennebec County, ME" & store_Type=="Supermarket Type1") %>% 
  select(total_sales))

#2
n=table(store_train$storecode)
dim(n)

#3
glimpse(store_train)
n=table(store_train$country)
dim(n)

#4
n=table(store_train$Areaname)
dim(n)
n

#5
n1=store_train %>% 
  filter(store_Type=="Grocery Store") %>% 
  count()
n1
n2=store_train %>% 
  filter(store_Type=="Grocery Store" & store==1) %>% 
  count()
n2
round((n2*100)/n1,2)
n3=store_train %>% 
  count()
round((n2*100)/n3,2)

#6
library(ggplot2)
shapiro.test(store_train$sales4)

ggplot(store_train, aes(x=sales4)) + 
  geom_histogram(binwidth=.25, colour="black", fill="white")

ggplot(store_train,aes(x=sales0))+geom_density(color="red")+
  geom_histogram(aes(y=..density..,alpha=0.5))+
  stat_function(fun=dnorm,aes(x=sales0),color="green")

ggplot(store_train, aes(x=sales0))+
  geom_histogram(aes(y=..density..),binwidth=.25,colour="black",fill="white")+
  stat_function(fun=dnorm,lwd=2,col='red',
                args=list(mean=mean(store_train$sales0),
                          sd=sd(store_train$sales0)))

#7
OutVals = boxplot(store_train$sales4)$out
length(which(store_train$sales4 %in% OutVals))

OutVals = boxplot(store_train$sales0, plot=FALSE)$out
length(OutVals)

IQR=IQR(store_train$sales0)
summary(store_train$sales0)
q1=summary(store_train$sales0)[["1st Qu."]]
q3=summary(store_train$sales0)[["3rd Qu."]]
lower=(q1-1.5*IQR)
upper=(q3+1.5*IQR)
n1=sum(as.numeric(store_train$sales0<lower))
n2=sum(as.numeric(store_train$sales0>upper))
n1+n2
141+168+136+159+59

n=store_train %>% 
  mutate(total_sales=sales0+sales1+sales2+sales3+sales4)
OutVals = boxplot(n$total_sales)$out
length(which(n$total_sales %in% OutVals))

#8
store_train %>% 
  mutate(total_sales=sales0+sales1+sales2+sales3+sales4) %>% 
  select(total_sales,store_Type) %>% 
  group_by(store_Type) %>% 
  summarise(var=var(total_sales)) %>% 
  filter(var==max(var))

#9
glimpse(store_train)
n1=table(store_train$state_alpha)
dim(n1)
sort(n1)
View(n1)

n2=store_train %>% 
  mutate(total_sales=sales0+sales1+sales2+sales3+sales4) %>% 
  group_by(state_alpha) %>% 
  summarise(avg=mean(total_sales)) %>% 
  select(state_alpha,avg)
View(n2)
min(n2$avg)
max(n2$avg)

n3=cbind(n1,n2)
View(n3)
sum(n3$Freq)


#10
class(store_train$store)

############################################################################