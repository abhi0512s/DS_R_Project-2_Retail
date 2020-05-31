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

for(col in names(store_all)){
  if(sum(is.na(store_all[,col]))>0 & (col %in% c("population","country"))){
    store_all[is.na(store_all[,col]),col]=round(mean(store_all[,col],na.rm=T))
  }
}

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

store_all=store_all %>% 
  select(-Id,-countyname,-storecode,-Areaname,-countytownname,-state_alpha)

store_train=store_all %>% filter(data=='train') %>% select(-data)
store_test=store_all %>% filter(data=='test') %>% select(-data,-store)

library(pROC)
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

num_trials=5

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

    best_params=params
  }
  print('DONE')
}

myauc

best_params

best_params=data.frame(interaction.depth=7,
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
                  n.trees = best_params$n.trees,type = "response")

write.table(test.pred,file ="Abhilash_Singh_P2_part2_GBM.csv",
            row.names = F,col.names="store")
