library(fds)
library(ggplot2)
library(reshape2)
library(MASS)
library(lattice)
library(fdapace)
library(refund)
library(Metrics)
library(grf)
library(e1071)
library(splines)
library(ggfortify)
library(neuralnet)
library(sigmoid)
library(splines2)
library(rlist)

#this file contains 3 network frameworks solved using fdnn and fbnn with EARLY STOPPING.

#value of hyperparameter S, vi, vk, vk, vn and vm don't affect the performance much.

# main parameter that needs to be tuned is the step size

#For large number of iteration, you might need MORE MEMORY.(Ideally run it on a cluster)



#defining relu activiation function
relud=function(aa){
  aa[which(aa>0)]=1
  aa[which(aa<0)]=0
  
  return(aa)
}



# We will use the fat spectrum as the dummy example.
x1=t(Fatspectrum$y)

m=length(x1[1,])
M=m
n=length(x1[,1])
N=n


x = seq(0,1,len=m)        #grid values


yy=(Fatvalues)


#trian
y=yy[1:176]
xt=x1[1:176,]


#es train
y=yy[(1:136)]
xt=x1[(1:136),]

#es validation
yv=yy[(137:176)]
xtv=x1[(137:176),]


#test
yp=yy[-(1:176)]
xtp=x1[-(1:176),]








####################################
#########FDNN
####################################
Y=y
X=xt

#Here we only have 1 hidden layer with 1 neuron.
#S is the grid length of the neuron in the hidden layer
#step_size is the step size of the gradient descent
#niteration is the number of iteration.

#X is a matrix containing our single predictor functions for the n samples measured at m timepoints each. We assume dense data.
# but code can be edited to take in more predictors and sparse/irregular data.
#Y is the response vector of size n.
# XV and YV are the validation set vlaues.


#the function below fits our network using FDNN strategy. It gives out the parameter values, loss and predict value.
fdnn.es <- function(X, Y, XV,YV, S, step_size, niteration){
  xv=XV
  yv=YV
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  
  # initialize parameters randomly
  W <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b <- matrix(0, nrow = N, ncol = S)
  W2 <- 0.01 * as.vector(rnorm(S))
  b2 <- matrix(0, nrow = N, ncol = 1)
  
  trainloss=c()
  testloss=c()
  
  Wlist=list()  
  blist=list()  
  W2list=list()  
  b2list=list()
  yhatlist=list()
  
  # gradient descent loop to update weight and bias
  for (i in 1:niteration){
    # hidden layer, ReLU activation
    
    hidden_layer1 <- relu(b+xt%*%W/M)
    hidden_layer1<- matrix(hidden_layer1, nrow  = N)
    # class score
    
    
    hidden_layer2 <- (b2+hidden_layer1%*%W2/S)
    hidden_layer2<- matrix(hidden_layer2, nrow  = N)
    
    
    # compute the loss: sofmax and regularization
    yhat=hidden_layer2
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    Nv <- nrow(xtv) # number of examples
    Mv <- ncol(xtv) # number of timepoints for xt
    hidden_layer1test <- relu(b[1:Nv,]+xv%*%W/Mv)
    hidden_layer1test<- matrix(hidden_layer1test, nrow  = Nv)
    # class score
    
    
    hidden_layer2test <- (b2[1:Nv]+hidden_layer1test%*%W2/S)
    hidden_layer2test<- matrix(hidden_layer2test, nrow  = Nv)
    
    
    # compute the loss: sofmax and regularization
    yhattest=hidden_layer2test
    data_losstest <- sum((yv-yhattest)^2)/N
    
    
    # check progress
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    dW2.1=dscores#*(as.matrix(hidden_layer2*(1-hidden_layer2)))
    
    dW2= t(hidden_layer1)%*%dW2.1
    
    db2=colSums(dW2.1)
    
    
    
    
    
    
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    
    dW1.1=(as.matrix(relud(hidden_layer1)))
    
    dW1.2= as.matrix(dW2.1)%*%t(as.matrix(W2))
    dW1.3=dW1.2*dW1.1
    dW1.4=t(xt)%*%dW1.3
    dW1=(dW1.4)
    
    
    db=colSums(dW1.3)
    
    
    # update parameter 
    W <- W-step_size*dW1
    b <- b[1,]-step_size*db
    b=matrix(rep(b,N),N,S, byrow = T)
    W2 <- W2-step_size*dW2
    b2 <- b2[1]-step_size*db2
    b2=rep(b2,N)
    
    trainloss=c(trainloss,data_loss)
    testloss=c(testloss,data_losstest)
    
    Wlist=list.append(Wlist,W)
    blist=list.append(blist,b)
    W2list=list.append(W2list,W2)
    b2list=list.append(b2list,b2)
    yhatlist=list.append(yhatlist,yhat)
    
  }
  kk=which.min(testloss)
  W=Wlist[[kk-1]]
  b=blist[[kk-1]]
  W2=W2list[[kk-1]]
  b2=b2list[[kk-1]]
  yhat=yhatlist[[kk-1]]
  loss=trainloss[kk]
  lossv=testloss[kk]
  
  return(list(W, W2, b, b2,sqrt(loss),yhat,trainloss,testloss,which.min(trainloss),which.min(testloss),kk,sqrt(lossv)))
}


#the function below fits the model learnt from fdnn function
#X is a matrix containing informaiton for the test data.
#S needs to be the same as specified in fdnn function.

fdnn.pred <- function(X, S, para = list()){
  W <- para[[1]]
  
  W2 <- para[[2]]
  
  
  b <- para[[3]]
  b=b[(1:length(X[,1])),]
  b2 <- para[[4]]
  b2=b2[(1:length(X[,1]))]
  
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  
  hidden_layer1 <- relu(b+xt%*%W/M)
  hidden_layer1<- matrix(hidden_layer1, nrow  = N)
  # class score
  
  
  hidden_layer2 <- (b2+hidden_layer1%*%W2/S)
  hidden_layer2<- matrix(hidden_layer2, nrow  = N)
  
  
  # compute the loss: sofmax and regularization
  yhat=hidden_layer2
  
  
  return(yhat)  
}


# running the model
#for best result you can lower the step size and increase the iteration vlaues depending on the memory space you have.
#Ideally run it on a cluster.
fdnn.model <- fdnn.es(X=xt, Y=Y,XV=xtv,YV=yv, S=30, step_size = 0.01, niteration = 5000)


#training loss
fdnn.pred.model <- fdnn.pred(X=xt, S=30, fdnn.model)
t44=rmse(as.numeric(y),as.numeric(fdnn.pred.model))

#Validation loss
fdnn.pred.model <- fdnn.pred(X=xtv, S=30, fdnn.model)
t44v=rmse(as.numeric(yv),as.numeric(fdnn.pred.model))

#testing error
fdnn.pred.model <- fdnn.pred(X=xtp, S=30, fdnn.model)
t4p4=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
t44
t44v
t4p4








####################################
#########FBNN
####################################


#Here we only have 1 hidden layer with 1 neuron.
#S is the grid length of the neuron in the hidden layer
#step_size is the step size of the gradient descent
#niteration is the number of iteration.

#X is a matrix containing our single predictor functions for the n samples measured at m timepoints each. We assume dense data.
# but code can be edited to take in more predictors and sparse/irregular data.
#Y is the response vector of size n.
# XV and YV are the validation set vlaues.


#vi, vj, vk are the number of basis functions (bspline) used in the hidden layers.

#the function below fits our network using FBNN strategy. It gives out the parameter values, loss and predict value.


Y=y
X=xt





#the function below fits our network using FBNN strategy. It gives out the parameter values, loss and predict value.
fbnn.es <- function(X, Y, XV,YV, S, vi, vj, vk, step_size, niteration){
  xv=XV
  yv=YV
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  
  # initialize parameters randomly
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  vk1 <- seq(0, 1, by=1/(S-1))
  vk_v <- t(as.matrix(bSpline(vk1,df=vk)))
  
  
  
  
  # initialize parameters randomly
  W <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b <- matrix(0, nrow = N, ncol = S)
  W2 <- 0.01 * as.vector(rnorm(vk))
  b2 <- matrix(0, nrow = N, ncol = 1)
  
  
  trainloss=c()
  testloss=c()
  
  Wlist=list()  
  blist=list()  
  W2list=list()  
  b2list=list()
  yhatlist=list()
  
  # gradient descent loop to update weight and bias
  for (i in 1:niteration){
    # hidden layer, ReLU activation
    
    
    ai=t(as.matrix(vi_v)%*%t(xt)/M)
    
    
    awv1=ai%*%W
    awv=awv1%*%vj_v
    
    hidden_layer1 <- relu(awv+b)
    hidden_layer1<- matrix(hidden_layer1, nrow  = N)
    # class score
    
    ak=t(as.matrix(vk_v)%*%t(hidden_layer1)/S)
    
    
    hl2=(ak%*%(as.vector(W2)))
    
    hidden_layer2 <- (b2+hl2)
    hidden_layer2<- matrix(hidden_layer2, nrow  = N)
    
    
    # compute the loss: sofmax and regularization
    yhat=hidden_layer2
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    
    
    
    ##validation set
    Nv <- nrow(xtv)
    aitest=t(as.matrix(vi_v)%*%t(xv)/M)
    
    
    awv1test=aitest%*%W
    awvtest=awv1test%*%vj_v
    
    hidden_layer1test <- relu(awvtest+b[1:Nv,])
    hidden_layer1test<- matrix(hidden_layer1test, nrow  = Nv)
    # class score
    
    aktest=t(as.matrix(vk_v)%*%t(hidden_layer1test)/S)
    
    
    hl2test=(aktest%*%(as.vector(W2)))
    
    hidden_layer2test <- (b2[1:Nv]+hl2test)
    hidden_layer2test<- matrix(hidden_layer2test, nrow  = Nv)
    
    
    # compute the loss: sofmax and regularization
    yhattest=hidden_layer2test
    data_losstest <- sum((yv-yhattest)^2)/N
    
    # check progress
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    dW2=t(ak)%*%dscores
    
    db2=colSums(dscores)
    
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    
    dW1.0=as.vector(W2)%*%vk_v
    dW1.1=dscores%*%dW1.0
    dW1.2=(as.matrix(relud(hidden_layer1)))
    dW1.3=dW1.1*dW1.2
    dW1.4=vj_v%*%t(dW1.3)
    dW1.5=dW1.4%*%(ai)
    dW1=t(dW1.5)   
    
    db=colSums(dW1.3)
    
    # update parameter
    W <- W-step_size*dW1
    b <- b[1,]-step_size*db
    b=matrix(rep(b,N),N,S, byrow = T)
    W2 <- W2-step_size*dW2
    b2 <- b2[1]-step_size*db2
    b2=rep(b2,N)
    
    trainloss=c(trainloss,data_loss)
    testloss=c(testloss,data_losstest)
    
    Wlist=list.append(Wlist,W)
    blist=list.append(blist,b)
    W2list=list.append(W2list,W2)
    b2list=list.append(b2list,b2)
    yhatlist=list.append(yhatlist,yhat)
    
  }
  kk=which.min(testloss)
  W=Wlist[[kk-1]]
  b=blist[[kk-1]]
  W2=W2list[[kk-1]]
  b2=b2list[[kk-1]]
  yhat=yhatlist[[kk-1]]
  loss=trainloss[kk]
  lossv=testloss[kk]
  
  return(list(W, W2, b, b2,sqrt(loss),yhat,trainloss,testloss,which.min(trainloss),which.min(testloss),kk,sqrt(lossv)))
}


#the function below fits the model learnt from fbnn function
#X is a matrix containing informaiton for the test data.
#S,vi,vj,vk needs to be the same as specified in fdnn function.
fbnn.pred <- function(X, S, vi, vj, vk, para = list()){
  W <- para[[1]]
  
  W2 <- para[[2]]
  b <- para[[3]]
  b=b[(1:length(X[,1])),]
  b2 <- para[[4]]
  b2=b2[(1:length(X[,1]))]
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  vk1 <- seq(0, 1, by=1/(S-1))
  vk_v <- t(as.matrix(bSpline(vk1,df=vk)))
  
  ai=t(as.matrix(vi_v)%*%t(xt)/M)
  
  
  awv1=ai%*%W
  awv=awv1%*%vj_v
  
  hidden_layer1 <- relu(awv+b)
  hidden_layer1<- matrix(hidden_layer1, nrow  = N)
  # class score
  
  ak=t(as.matrix(vk_v)%*%t(hidden_layer1)/S)
  
  
  hl2=(ak%*%(as.vector(W2)))
  
  hidden_layer2 <- (b2+hl2)
  hidden_layer2<- matrix(hidden_layer2, nrow  = N)
  
  
  # compute the loss: sofmax and regularization
  yhat=hidden_layer2
  
  return(yhat)  
}


fbnn.model <- fbnn.es(X=xt, Y=Y,XV=xtv,YV=yv, S=30, vi=5, vj=5, vk=5, step_size=.01, niteration = 5000)


#training error
t4_i2=rmse(as.numeric(y),as.numeric(fbnn.pred(X=xt, S=30, vi=5, vj=5, vk=5, fbnn.model)))
t4_i2

#Validation loss
fbnn.pred.model <- fbnn.pred(X=xtv, S=30, vi=5, vj=5, vk=5, fbnn.model)
t4p_i2v=rmse(as.numeric(yv),as.numeric(fbnn.pred.model))

#testing error
fbnn.pred.model <- fbnn.pred(X=xtp, S=30, vi=5, vj=5, vk=5, fbnn.model)
t4p_i2=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
t4_i2
t4p_i2v
t4p_i2











####################################
#########FdNN22
####################################


#Here we are defining 2 hidden layers with 2 neurons each and using fdnn strategy
fdnn2.2.es <- function(X, Y,XV,YV, S, step_size, niteration){
  xv=XV
  yv=YV
  xt=X
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  
  # initialize parameters randomly
  W1_1 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_1 <- matrix(0, nrow = N, ncol = S)
  W1_2 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_2 <- matrix(0, nrow = N, ncol = S)
  
  W2_11 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_12 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_21 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_22 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  b2_1 <- matrix(0, nrow = N, ncol = S)
  b2_2 <- matrix(0, nrow = N, ncol = S)
  
  #second layer
  b3_1 <- matrix(0, nrow = N, ncol = 1)
  W3_1 <- 0.01 * as.vector(rnorm(S))
  W3_2 <- 0.01 * as.vector(rnorm(S))
  
  
  trainloss=c()
  testloss=c()
  
  W1_1list=list()  
  b1_1list=list()  
  W1_2list=list()  
  b1_2list=list()
  
  W2_11list=list()
  W2_12list=list()
  W2_21list=list()
  W2_22list=list()
  b2_1list=list()
  b2_2list=list()
  
  W3_1list=list()   
  W3_2list=list()  
  b3_1list=list()  
  
  yhatlist=list()
  
  
  # gradient descent loop to update weight and bias
  for (i in 0:niteration){
    # hidden layer, ReLU activation
    
    hidden_layer11 <- relu(b1_1+xt%*%W1_1/M)
    hidden_layer11 <- matrix(hidden_layer11, nrow  = N)
    
    
    hidden_layer12 <- relu(b1_2+xt%*%W1_2/M)
    hidden_layer12 <- matrix(hidden_layer12, nrow  = N)
    
    # class score
    
    
    hidden_layer21 <- relu(b2_1+(hidden_layer11%*%W2_11/S)+(hidden_layer12%*%W2_21/S))
    hidden_layer21 <- matrix(hidden_layer21, nrow  = N)
    
    hidden_layer22 <- relu(b2_2+(hidden_layer11%*%W2_12/S)+(hidden_layer12%*%W2_22/S))
    hidden_layer22 <- matrix(hidden_layer22, nrow  = N)
    
    
    # compute the loss: sofmax and regularization
    yhat=(b3_1+(hidden_layer21%*%W3_1/S)+(hidden_layer22%*%W3_2/S))
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    
    #validation
    Nv <- nrow(xtv)
    hidden_layer11test <- relu(b1_1[1:Nv,]+xv%*%W1_1/M)
    hidden_layer11test <- matrix(hidden_layer11test, nrow  = Nv)
    
    
    hidden_layer12test <- relu(b1_2[1:Nv,]+xv%*%W1_2/M)
    hidden_layer12test <- matrix(hidden_layer12test, nrow  = Nv)
    
    # class score
    
    
    hidden_layer21test <- relu(b2_1[1:Nv,]+(hidden_layer11test%*%W2_11/S)+(hidden_layer12test%*%W2_21/S))
    hidden_layer21test <- matrix(hidden_layer21test, nrow  = Nv)
    
    hidden_layer22test <- relu(b2_2[1:Nv,]+(hidden_layer11test%*%W2_12/S)+(hidden_layer12test%*%W2_22/S))
    hidden_layer22test <- matrix(hidden_layer22test, nrow  = Nv)
    
    
    # compute the loss: sofmax and regularization
    yhattest=(b3_1[1:Nv]+(hidden_layer21test%*%W3_1/S)+(hidden_layer22test%*%W3_2/S))
    data_losstest <- sum((yv-yhattest)^2)/N
    losstest <- data_losstest
    
    
    
    
    
    # check progress
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    
    dW3_10=dscores
    
    dW3_1= t(hidden_layer21)%*%dW3_10
    
    db3_1=colSums(dW3_10)
    
    dW3_2= t(hidden_layer22)%*%dW3_10
    
    
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    dscores= -2*(Y-yhat)/N
    
    #dstep0=t(as.matrix(hidden_layer21*(1-hidden_layer21)))
    dW2_11.1=(as.matrix(relud(hidden_layer21)))
    dW2_11.2=as.matrix(dscores)%*%t(as.matrix(W3_1))
    dW2_11.3=dW2_11.2*dW2_11.1
    dW2_11.4=t(hidden_layer11)%*%dW2_11.3
    dW2_11=(dW2_11.4)
    
    
    db2_1=colSums(dW2_11.3)
    
    
    dW2_21.4=t(hidden_layer12)%*%dW2_11.3
    dW2_21=(dW2_21.4)
    
    
    
    dW2_12.1=(as.matrix(relud(hidden_layer22)))
    dW2_12.2=as.matrix(dscores)%*%t(as.matrix(W3_2))
    dW2_12.3=dW2_12.2*dW2_12.1
    dW2_12.4=t(hidden_layer11)%*%dW2_12.3
    dW2_12=(dW2_12.4)
    
    
    db2_2=colSums(dW2_12.3)
    
    
    dW2_22.4=t(hidden_layer12)%*%dW2_12.3
    dW2_22=(dW2_21.4)
    
    
    
    
    
    ########################
    dW1_1.1=dW2_11.3%*%t(W2_11)+dW2_12.3%*%t(W2_12)
    dW1_1.2=(as.matrix(relud(hidden_layer11)))
    dW1_1.3=dW1_1.1*dW1_1.2
    dW1_1.4=t(xt)%*%dW1_1.3
    dW1_1=dW1_1.4
    
    db1_1=colSums(dW1_1.3)
    
    
    dW1_2.1=dW2_11.3%*%t(W2_21)+dW2_12.3%*%t(W2_22)
    dW1_2.2=(as.matrix(relud(hidden_layer12)))
    dW1_2.3=dW1_2.1*dW1_2.2
    dW1_2.4=t(xt)%*%dW1_2.3
    dW1_2=dW1_2.4
    
    db1_2=colSums(dW1_2.3)
    
    # update parameter 
    
    W1_1=W1_1-step_size*dW1_1
    b1_1<- b1_1[1,]-step_size*db1_1
    b1_1=matrix(rep(b1_1,N),N,S, byrow = T)
    W1_2=W1_2-step_size*dW1_2
    b1_2<- b1_2[1,]-step_size*db1_2
    b1_2=matrix(rep(b1_2,N),N,S, byrow = T)
    
    
    W2_11 <- W2_11-step_size*dW2_11
    W2_21 <- W2_21-step_size*dW2_21
    W2_12 <- W2_12-step_size*dW2_12
    W2_22 <- W2_22-step_size*dW2_22
    b2_1<- b2_1[1,]-step_size*db2_1
    b2_1=matrix(rep(b2_1,N),N,S, byrow = T)
    b2_2<- b2_2[1,]-step_size*db2_2
    b2_2=matrix(rep(b2_2,N),N,S, byrow = T)
    
    W3_1 <- W3_1-step_size*dW3_1
    W3_2 <- W3_2-step_size*dW3_2    
    b3_1 <- b3_1[1]-step_size*db3_1
    b3_1=rep(b3_1,N)
    
    
    trainloss=c(trainloss,data_loss)
    testloss=c(testloss,data_losstest)
    
    W1_1list=list.append(W1_1list,W1_1)  
    b1_1list=list.append(b1_1list,b1_1)  
    W1_2list=list.append(W1_2list,W1_2)  
    b1_2list=list.append(b1_2list,b1_2)
    
    W2_11list=list.append(W2_11list,W2_11)
    W2_12list=list.append(W2_12list,W2_12)
    W2_21list=list.append(W2_21list,W2_21)
    W2_22list=list.append(W2_22list,W2_22)
    b2_1list=list.append(b2_1list,b2_1)
    b2_2list=list.append(b2_2list,b2_2)
    
    W3_1list=list.append(W3_1list,W3_1)   
    W3_2list=list.append(W3_2list,W3_2)  
    b3_1list=list.append(b3_1list,b3_1) 
    
    
    yhatlist=list.append(yhatlist,yhat)
    
    
    
  }
  kk=which.min(testloss)
  
  W1_1 <-W1_1list[[kk-1]]
  b1_1 <-b1_1list[[kk-1]]
  W1_2 <- W1_2list[[kk-1]]
  b1_2 <-b1_2list[[kk-1]]
  
  W2_11 <-W2_11list[[kk-1]]
  W2_12 <-W2_12list[[kk-1]]
  W2_21 <- W2_21list[[kk-1]]
  W2_22 <- W2_22list[[kk-1]]
  b2_1 <- b2_1list[[kk-1]]
  b2_2 <- b2_2list[[kk-1]]
  
  #second layer
  b3_1 <-b3_1list[[kk-1]]
  W3_1 <-W3_1list[[kk-1]]
  W3_2 <- W3_2list[[kk-1]]
  
  
  yhat=yhatlist[[kk-1]]
  loss=trainloss[kk]
  lossv=testloss[kk]
  
  return(list(W1_1, W1_2, b1_1, b1_2, 
              W2_11, W2_12, W2_21, W2_22, b2_1, b2_2, 
              W3_1, W3_2, b3_1,
              sqrt(loss),yhat,trainloss,testloss,which.min(trainloss),which.min(testloss),kk,sqrt(lossv)))
}




fdnn.pred2.2 <- function(X, S, para = list()){
  
  
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  
  W1_1 <-para[[1]]
  W1_2 <-para[[2]]
  
  
  b1_1 <-para[[3]]
  b1_1=b1_1[(1:length(X[,1])),]
  b1_2 <-para[[4]]
  b1_2=b1_2[(1:length(X[,1])),]
  
  
  #second layer
  
  W2_11 <-para[[5]]
  W2_12 <-para[[6]]
  W2_21 <-para[[7]]
  W2_22 <-para[[8]]
  
  
  b2_1 <-para[[9]]
  b2_1=b2_1[(1:length(X[,1])),]
  
  b2_2 <-para[[10]]
  b2_2=b2_2[(1:length(X[,1])),]
  
  
  
  W3_1 <-para[[11]]
  W3_2 <-para[[12]]
  
  b3_1 <-para[[13]]
  b3_1=b3_1[(1:length(X[,1]))]
  
  
  hidden_layer11 <- relu(b1_1+xt%*%W1_1/M)
  hidden_layer11 <- matrix(hidden_layer11, nrow  = N)
  
  
  hidden_layer12 <- relu(b1_2+xt%*%W1_2/M)
  hidden_layer12 <- matrix(hidden_layer12, nrow  = N)
  
  # class score
  
  
  hidden_layer21 <- relu(b2_1+(hidden_layer11%*%W2_11/S)+(hidden_layer12%*%W2_21/S))
  hidden_layer21 <- matrix(hidden_layer21, nrow  = N)
  
  hidden_layer22 <- relu(b2_2+(hidden_layer11%*%W2_12/S)+(hidden_layer12%*%W2_22/S))
  hidden_layer22 <- matrix(hidden_layer22, nrow  = N)
  
  
  # compute the loss: sofmax and regularization
  yhat=(b3_1+(hidden_layer21%*%W3_1/S)+(hidden_layer22%*%W3_2/S))
  
  
  
  return(yhat)  
}




fdnn.model <- fdnn2.2.es(X=xt, Y=Y,XV=xtv,YV=yv, S=30, step_size = 0.01, niteration = 5000)

t44=rmse(as.numeric(y),as.numeric(fdnn.pred2.2(X=xt, S=30, fdnn.model)))
t44

fdnn.pred.model <- fdnn.pred2.2(X=xtv, S=30, fdnn.model)
t4p4v=rmse(as.numeric(yv),as.numeric(fdnn.pred.model))

fdnn.pred.model <- fdnn.pred2.2(X=xtp, S=30, fdnn.model)
t4p4=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
t44
t4p4v
t4p4





####################################
#########FdNN44
####################################


#Here we are defining 2 hidden layers with 4 neurons each and using fdnn strategy

fdnn4.4.es <- function(X, Y,XV,YV, S, step_size, niteration){
  xv=XV
  yv=YV
  xt=X
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  
  # initialize parameters randomly
  W1_1 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_1 <- matrix(0, nrow = N, ncol = S)
  W1_2 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_2 <- matrix(0, nrow = N, ncol = S)
  
  W1_3 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_3 <- matrix(0, nrow = N, ncol = S)
  W1_4 <- 0.01 * matrix(rnorm(M*S), nrow = M)
  b1_4 <- matrix(0, nrow = N, ncol = S)
  
  W2_11 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_21 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_31 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_41 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  W2_12 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_22 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_32 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_42 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  W2_13 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_23 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_33 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_43 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  W2_14 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_24 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_34 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  W2_44 <- 0.01 * matrix(rnorm(S*S), nrow = S)
  
  
  
  
  b2_1 <- matrix(0, nrow = N, ncol = S)
  b2_2 <- matrix(0, nrow = N, ncol = S)
  b2_3 <- matrix(0, nrow = N, ncol = S)
  b2_4 <- matrix(0, nrow = N, ncol = S)
  
  #second layer
  b3_1 <- matrix(0, nrow = N, ncol = 1)
  
  W3_1 <- 0.01 * as.vector(rnorm(S))
  W3_2 <- 0.01 * as.vector(rnorm(S))
  W3_3 <- 0.01 * as.vector(rnorm(S))
  W3_4 <- 0.01 * as.vector(rnorm(S))
  
  
  
  
  trainloss=c()
  testloss=c()
  
  
  W1_1list=list()  
  b1_1list=list()  
  W1_2list=list()  
  b1_2list=list()
  
  W1_3list=list()  
  b1_3list=list()  
  W1_4list=list()  
  b1_4list=list()
  
  W2_11list=list()
  W2_21list=list()
  W2_31list=list()
  W2_41list=list()
  
  W2_12list=list()
  W2_22list=list()
  W2_32list=list()
  W2_42list=list()
  
  W2_13list=list()
  W2_23list=list()
  W2_33list=list()
  W2_43list=list()
  
  W2_14list=list()
  W2_24list=list()
  W2_34list=list()
  W2_44list=list()
  
  b2_1list=list()
  b2_2list=list()
  b2_3list=list()
  b2_4list=list()
  
  
  
  W3_1list=list()
  W3_2list=list()
  W3_3list=list()
  W3_4list=list()
  
  b3_1list=list()
  
  yhatlist=list()
  
  
  # gradient descent loop to update weight and bias
  for (i in 0:niteration){
    # hidden layer, ReLU activation
    
    hidden_layer11 <- relu(b1_1+xt%*%W1_1/M)
    hidden_layer11 <- matrix(hidden_layer11, nrow  = N)
    
    
    hidden_layer12 <- relu(b1_2+xt%*%W1_2/M)
    hidden_layer12 <- matrix(hidden_layer12, nrow  = N)
    
    
    hidden_layer13 <- relu(b1_3+xt%*%W1_3/M)
    hidden_layer13 <- matrix(hidden_layer13, nrow  = N)
    
    
    hidden_layer14 <- relu(b1_4+xt%*%W1_4/M)
    hidden_layer14 <- matrix(hidden_layer14, nrow  = N)
    
    # class score
    
    
    hidden_layer21 <- relu(b2_1+(hidden_layer11%*%W2_11/S)+(hidden_layer12%*%W2_21/S)+(hidden_layer13%*%W2_31/S)+(hidden_layer14%*%W2_41/S))
    hidden_layer21 <- matrix(hidden_layer21, nrow  = N)
    
    hidden_layer22 <- relu(b2_2+(hidden_layer11%*%W2_12/S)+(hidden_layer12%*%W2_22/S)+(hidden_layer13%*%W2_32/S)+(hidden_layer14%*%W2_42/S))
    hidden_layer22 <- matrix(hidden_layer22, nrow  = N)
    
    hidden_layer23 <- relu(b2_3+(hidden_layer11%*%W2_13/S)+(hidden_layer12%*%W2_23/S)+(hidden_layer13%*%W2_33/S)+(hidden_layer14%*%W2_43/S))
    hidden_layer23 <- matrix(hidden_layer23, nrow  = N)
    
    hidden_layer24 <- relu(b2_4+(hidden_layer11%*%W2_14/S)+(hidden_layer12%*%W2_24/S)+(hidden_layer13%*%W2_34/S)+(hidden_layer14%*%W2_44/S))
    hidden_layer24 <- matrix(hidden_layer24, nrow  = N)
    
    
    # compute the loss: sofmax and regularization
    yhat=(b3_1+(hidden_layer21%*%W3_1/S)+(hidden_layer22%*%W3_2/S)+(hidden_layer23%*%W3_3/S)+(hidden_layer24%*%W3_4/S))
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    
    #validation
    Nv <- nrow(xtv)
    
    hidden_layer11test <- relu(b1_1[1:Nv,]+xv%*%W1_1/M)
    hidden_layer11test <- matrix(hidden_layer11test, nrow  = Nv)
    
    
    hidden_layer12test <- relu(b1_2[1:Nv,]+xv%*%W1_2/M)
    hidden_layer12test <- matrix(hidden_layer12test, nrow  = Nv)
    
    
    hidden_layer13test <- relu(b1_3[1:Nv,]+xv%*%W1_3/M)
    hidden_layer13test <- matrix(hidden_layer13test, nrow  = Nv)
    
    
    hidden_layer14test <- relu(b1_4[1:Nv,]+xv%*%W1_4/M)
    hidden_layer14test <- matrix(hidden_layer14test, nrow  = Nv)
    
    # class score
    
    
    hidden_layer21test <- relu(b2_1[1:Nv,]+(hidden_layer11test%*%W2_11/S)+(hidden_layer12test%*%W2_21/S)+
                                 (hidden_layer13test%*%W2_31/S)+(hidden_layer14test%*%W2_41/S))
    hidden_layer21test <- matrix(hidden_layer21test, nrow  = Nv)
    
    hidden_layer22test <- relu(b2_2[1:Nv,]+(hidden_layer11test%*%W2_12/S)+(hidden_layer12test%*%W2_22/S)+
                                 (hidden_layer13test%*%W2_32/S)+(hidden_layer14test%*%W2_42/S))
    hidden_layer22test <- matrix(hidden_layer22test, nrow  = Nv)
    
    hidden_layer23test <- relu(b2_3[1:Nv,]+(hidden_layer11test%*%W2_13/S)+(hidden_layer12test%*%W2_23/S)+
                                 (hidden_layer13test%*%W2_33/S)+(hidden_layer14test%*%W2_43/S))
    hidden_layer23test <- matrix(hidden_layer23test, nrow  = Nv)
    
    hidden_layer24test <- relu(b2_4[1:Nv,]+(hidden_layer11test%*%W2_14/S)+(hidden_layer12test%*%W2_24/S)+
                                 (hidden_layer13test%*%W2_34/S)+(hidden_layer14test%*%W2_44/S))
    hidden_layer24test <- matrix(hidden_layer24test, nrow  = Nv)
    
    
    # compute the loss: sofmax and regularization
    yhattest=(b3_1[1:Nv]+(hidden_layer21test%*%W3_1/S)+(hidden_layer22test%*%W3_2/S)+
                (hidden_layer23test%*%W3_3/S)+(hidden_layer24test%*%W3_4/S))
    data_losstest <- sum((yv-yhattest)^2)/N
    losstest <- data_losstest
    
    
    # check progress
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    
    dW3_10=dscores
    
    dW3_1= t(hidden_layer21)%*%dW3_10
    
    db3_1=colSums(dW3_10)
    
    dW3_2= t(hidden_layer22)%*%dW3_10
    
    dW3_3= t(hidden_layer23)%*%dW3_10
    
    dW3_4= t(hidden_layer24)%*%dW3_10
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    dscores= -2*(Y-yhat)/N
    
    #dstep0=t(as.matrix(hidden_layer21*(1-hidden_layer21)))
    dW2_11.1=(as.matrix(relud(hidden_layer21)))
    dW2_11.2=as.matrix(dscores)%*%t(as.matrix(W3_1))
    dW2_11.3=dW2_11.2*dW2_11.1
    dW2_11.4=t(hidden_layer11)%*%dW2_11.3
    dW2_11=(dW2_11.4)
    
    
    db2_1=colSums(dW2_11.3)
    
    
    dW2_21.4=t(hidden_layer12)%*%dW2_11.3
    dW2_21=(dW2_21.4)    
    dW2_31.4=t(hidden_layer13)%*%dW2_11.3
    dW2_31=(dW2_31.4)    
    dW2_41.4=t(hidden_layer14)%*%dW2_11.3
    dW2_41=(dW2_41.4)
    
    
    
    
    dW2_12.1=(as.matrix(relud(hidden_layer22)))
    dW2_12.2=as.matrix(dscores)%*%t(as.matrix(W3_2))
    dW2_12.3=dW2_12.2*dW2_12.1
    dW2_12.4=t(hidden_layer11)%*%dW2_12.3
    dW2_12=(dW2_12.4)
    
    
    db2_2=colSums(dW2_12.3)
    
    
    dW2_22.4=t(hidden_layer12)%*%dW2_12.3
    dW2_22=(dW2_22.4)    
    dW2_32.4=t(hidden_layer13)%*%dW2_12.3
    dW2_32=(dW2_32.4)    
    dW2_42.4=t(hidden_layer14)%*%dW2_12.3
    dW2_42=(dW2_42.4)
    
    
    
    
    dW2_13.1=(as.matrix(relud(hidden_layer23)))
    dW2_13.2=as.matrix(dscores)%*%t(as.matrix(W3_3))
    dW2_13.3=dW2_13.2*dW2_13.1
    dW2_13.4=t(hidden_layer11)%*%dW2_13.3
    dW2_13=(dW2_13.4)
    
    
    db2_3=colSums(dW2_13.3)
    
    
    dW2_23.4=t(hidden_layer12)%*%dW2_13.3
    dW2_23=(dW2_23.4)    
    dW2_33.4=t(hidden_layer13)%*%dW2_13.3
    dW2_33=(dW2_33.4)    
    dW2_43.4=t(hidden_layer14)%*%dW2_13.3
    dW2_43=(dW2_43.4)
    
    
    
    dW2_14.1=(as.matrix(relud(hidden_layer24)))
    dW2_14.2=as.matrix(dscores)%*%t(as.matrix(W3_4))
    dW2_14.3=dW2_14.2*dW2_14.1
    dW2_14.4=t(hidden_layer11)%*%dW2_14.3
    dW2_14=(dW2_14.4)
    
    
    db2_4=colSums(dW2_14.3)
    
    
    dW2_24.4=t(hidden_layer12)%*%dW2_14.3
    dW2_24=(dW2_24.4)    
    dW2_34.4=t(hidden_layer13)%*%dW2_14.3
    dW2_34=(dW2_34.4)    
    dW2_44.4=t(hidden_layer14)%*%dW2_14.3
    dW2_44=(dW2_44.4)
    
    
    
    
    ########################
    dW1_1.1=dW2_11.3%*%t(W2_11)+dW2_12.3%*%t(W2_12)+dW2_13.3%*%t(W2_13)+dW2_14.3%*%t(W2_14)
    dW1_1.2=(as.matrix(relud(hidden_layer11)))
    dW1_1.3=dW1_1.1*dW1_1.2
    dW1_1.4=t(xt)%*%dW1_1.3
    dW1_1=dW1_1.4
    
    db1_1=colSums(dW1_1.3)
    
    
    dW1_2.1=dW2_11.3%*%t(W2_21)+dW2_12.3%*%t(W2_22)+dW2_13.3%*%t(W2_23)+dW2_14.3%*%t(W2_24)
    dW1_2.2=(as.matrix(relud(hidden_layer12)))
    dW1_2.3=dW1_2.1*dW1_2.2
    dW1_2.4=t(xt)%*%dW1_2.3
    dW1_2=dW1_2.4
    
    db1_2=colSums(dW1_2.3)
    
    
    
    dW1_3.1=dW2_11.3%*%t(W2_31)+dW2_12.3%*%t(W2_32)+dW2_13.3%*%t(W2_33)+dW2_14.3%*%t(W2_34)
    dW1_3.2=(as.matrix(relud(hidden_layer13)))
    dW1_3.3=dW1_3.1*dW1_3.2
    dW1_3.4=t(xt)%*%dW1_3.3
    dW1_3=dW1_1.4
    
    db1_3=colSums(dW1_3.3)
    
    
    dW1_4.1=dW2_11.3%*%t(W2_41)+dW2_12.3%*%t(W2_42)+dW2_13.3%*%t(W2_43)+dW2_14.3%*%t(W2_44)
    dW1_4.2=(as.matrix(relud(hidden_layer14)))
    dW1_4.3=dW1_4.1*dW1_4.2
    dW1_4.4=t(xt)%*%dW1_4.3
    dW1_4=dW1_4.4
    
    db1_4=colSums(dW1_4.3)
    
    # update parameter 
    
    W1_1=W1_1-step_size*dW1_1
    b1_1<- b1_1[1,]-step_size*db1_1
    b1_1=matrix(rep(b1_1,N),N,S, byrow = T)
    W1_2=W1_2-step_size*dW1_2
    b1_2<- b1_2[1,]-step_size*db1_2
    b1_2=matrix(rep(b1_2,N),N,S, byrow = T)
    W1_3=W1_3-step_size*dW1_3
    b1_3<- b1_3[1,]-step_size*db1_3
    b1_3=matrix(rep(b1_3,N),N,S, byrow = T)
    W1_4=W1_4-step_size*dW1_4
    b1_4<- b1_4[1,]-step_size*db1_4
    b1_4=matrix(rep(b1_4,N),N,S, byrow = T)
    
    W2_11 <- W2_11-step_size*dW2_11
    W2_12 <- W2_12-step_size*dW2_12
    W2_13 <- W2_13-step_size*dW2_13
    W2_14 <- W2_14-step_size*dW2_14 
    W2_21 <- W2_21-step_size*dW2_21
    W2_22 <- W2_22-step_size*dW2_22
    W2_23 <- W2_23-step_size*dW2_23
    W2_24 <- W2_24-step_size*dW2_24    
    W2_31 <- W2_31-step_size*dW2_31
    W2_32 <- W2_32-step_size*dW2_32
    W2_33 <- W2_33-step_size*dW2_33
    W2_34 <- W2_34-step_size*dW2_34    
    W2_41 <- W2_41-step_size*dW2_41
    W2_42 <- W2_42-step_size*dW2_42
    W2_43 <- W2_43-step_size*dW2_43
    W2_44 <- W2_44-step_size*dW2_44
    
    
    b2_1<- b2_1[1,]-step_size*db2_1
    b2_1=matrix(rep(b2_1,N),N,S, byrow = T)
    b2_2<- b2_2[1,]-step_size*db2_2
    b2_2=matrix(rep(b2_2,N),N,S, byrow = T)
    b2_3<- b2_3[1,]-step_size*db2_3
    b2_3=matrix(rep(b2_3,N),N,S, byrow = T)
    b2_4<- b2_4[1,]-step_size*db2_4
    b2_4=matrix(rep(b2_4,N),N,S, byrow = T)
    
    W3_1 <- W3_1-step_size*dW3_1
    W3_2 <- W3_2-step_size*dW3_2   
    W3_3 <- W3_3-step_size*dW3_3
    W3_4 <- W3_4-step_size*dW3_4
    
    b3_1 <- b3_1[1]-step_size*db3_1
    b3_1=rep(b3_1,N)    
    
    trainloss=c(trainloss,data_loss)
    testloss=c(testloss,data_losstest)
    
    
    
    W1_1list=list.append(W1_1list,W1_1)  
    b1_1list=list.append(b1_1list,b1_1)  
    W1_2list=list.append(W1_2list,W1_2)  
    b1_2list=list.append(b1_2list,b1_2)
    
    W1_3list=list.append(W1_3list,W1_3)  
    b1_3list=list.append(b1_3list,b1_3)  
    W1_4list=list.append(W1_4list,W1_4)  
    b1_4list=list.append(b1_4list,b1_4)
    
    W2_11list=list.append(W2_11list,W2_11)
    W2_21list=list.append(W2_21list,W2_21)
    W2_31list=list.append(W2_31list,W2_31)
    W2_41list=list.append(W2_41list,W2_41)
    
    W2_12list=list.append(W2_12list,W2_12)
    W2_22list=list.append(W2_22list,W2_22)
    W2_32list=list.append(W2_32list,W2_32)
    W2_42list=list.append(W2_42list,W2_42)
    
    W2_13list=list.append(W2_13list,W2_13)
    W2_23list=list.append(W2_23list,W2_23)
    W2_33list=list.append(W2_33list,W2_33)
    W2_43list=list.append(W2_43list,W2_43)
    
    W2_14list=list.append(W2_14list,W2_14)
    W2_24list=list.append(W2_24list,W2_24)
    W2_34list=list.append(W2_34list,W2_34)
    W2_44list=list.append(W2_44list,W2_44)
    
    b2_1list=list.append(b2_1list,b2_1)
    b2_2list=list.append(b2_2list,b2_2)
    b2_3list=list.append(b2_3list,b2_3)
    b2_4list=list.append(b2_4list,b2_4)
    
    
    
    W3_1list=list.append(W3_1list,W3_1)
    W3_2list=list.append(W3_2list,W3_2)
    W3_3list=list.append(W3_3list,W3_3)
    W3_4list=list.append(W3_4list,W3_4)
    
    b3_1list=list.append(b3_1list,b3_1)
    
    
    yhatlist=list.append(yhatlist,yhat)
    
    
    
    
  }
  kk=which.min(testloss)
  
  W1_1 <-W1_1list[[kk-1]]
  b1_1 <-b1_1list[[kk-1]]
  W1_2 <- W1_2list[[kk-1]]
  b1_2 <-b1_2list[[kk-1]]
  
  
  W1_3 <-W1_3list[[kk-1]]
  b1_3 <-b1_3list[[kk-1]]
  W1_4 <- W1_4list[[kk-1]]
  b1_4 <-b1_4list[[kk-1]]
  
  W2_11 <-W2_11list[[kk-1]]
  W2_21 <-W2_21list[[kk-1]]
  W2_31 <-W2_31list[[kk-1]]
  W2_41 <-W2_41list[[kk-1]]
  
  W2_12 <-W2_12list[[kk-1]]
  W2_22 <-W2_22list[[kk-1]]
  W2_32 <-W2_32list[[kk-1]]
  W2_42 <-W2_42list[[kk-1]]
  
  W2_13 <-W2_13list[[kk-1]]
  W2_23 <-W2_23list[[kk-1]]
  W2_33 <-W2_33list[[kk-1]]
  W2_43 <-W2_43list[[kk-1]]
  
  W2_14 <-W2_14list[[kk-1]]
  W2_24 <-W2_24list[[kk-1]]
  W2_34 <-W2_34list[[kk-1]]
  W2_44 <-W2_44list[[kk-1]]
  
  b2_1 <-b2_1list[[kk-1]]
  b2_2 <-b2_2list[[kk-1]]
  b2_3 <-b2_3list[[kk-1]]
  b2_4 <-b2_4list[[kk-1]]
  #second layer
  
  W3_1 <- W3_1list[[kk-1]]
  W3_2 <- W3_2list[[kk-1]]
  W3_3 <- W3_3list[[kk-1]]
  W3_4 <- W3_4list[[kk-1]]
  
  b3_1 <-b3_1list[[kk-1]]
  
  
  yhat=yhatlist[[kk-1]]
  loss=trainloss[kk]
  lossv=testloss[kk]
  
  return(list(W1_1, W1_2, W1_3, W1_4, b1_1, b1_2, b1_3, b1_4, 
              W2_11, W2_12, W2_13, W2_14, W2_21, W2_22, W2_23, W2_24, W2_31, W2_32, W2_33, W2_34, W2_41, W2_42, W2_43, W2_44,
              b2_1, b2_2, b2_3, b2_4, 
              W3_1, W3_2, W3_3, W3_4, b3_1,
              sqrt(loss),yhat,trainloss,testloss,which.min(trainloss),which.min(testloss),kk,sqrt(lossv)))
}


fdnn.pred4.4 <- function(X, S, para = list()){
  
  
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  
  W1_1 <-para[[1]]
  W1_2 <-para[[2]]  
  W1_3 <-para[[3]]
  W1_4 <-para[[4]]
  
  
  b1_1 <-para[[5]]
  b1_2 <-para[[6]]  
  b1_3 <-para[[7]]
  b1_4 <-para[[8]]
  
  b1_1=b1_1[(1:length(X[,1])),]
  b1_2=b1_2[(1:length(X[,1])),]
  b1_3=b1_3[(1:length(X[,1])),]
  b1_4=b1_4[(1:length(X[,1])),]
  
  
  #second layer
  
  W2_11 <-para[[9]]
  W2_12 <-para[[10]]
  W2_13 <-para[[11]]
  W2_14 <-para[[12]]  
  W2_21 <-para[[13]]
  W2_22 <-para[[14]]
  W2_23 <-para[[15]]
  W2_24 <-para[[16]]  
  W2_31 <-para[[17]]
  W2_32 <-para[[18]]
  W2_33 <-para[[19]]
  W2_34 <-para[[20]]  
  W2_41 <-para[[21]]
  W2_42 <-para[[22]]
  W2_43 <-para[[23]]
  W2_44 <-para[[24]]
  
  
  b2_1 <-para[[25]]
  b2_2 <-para[[26]]  
  b2_3 <-para[[27]]
  b2_4 <-para[[28]]
  
  b2_1=b2_1[(1:length(X[,1])),]
  b2_2=b2_2[(1:length(X[,1])),]
  b2_3=b2_3[(1:length(X[,1])),]
  b2_4=b2_4[(1:length(X[,1])),]
  
  
  W3_1 <-para[[29]]
  W3_2 <-para[[30]]  
  W3_3 <-para[[31]]
  W3_4 <-para[[32]]
  
  b3_1 <-para[[33]]
  b3_1=b3_1[(1:length(X[,1]))]
  
  
  
  hidden_layer11 <- relu(b1_1+xt%*%W1_1/M)
  hidden_layer11 <- matrix(hidden_layer11, nrow  = N)
  
  
  hidden_layer12 <- relu(b1_2+xt%*%W1_2/M)
  hidden_layer12 <- matrix(hidden_layer12, nrow  = N)
  
  
  hidden_layer13 <- relu(b1_3+xt%*%W1_3/M)
  hidden_layer13 <- matrix(hidden_layer13, nrow  = N)
  
  
  hidden_layer14 <- relu(b1_4+xt%*%W1_4/M)
  hidden_layer14 <- matrix(hidden_layer14, nrow  = N)
  
  # class score
  
  
  hidden_layer21 <- relu(b2_1+(hidden_layer11%*%W2_11/S)+(hidden_layer12%*%W2_21/S)+(hidden_layer13%*%W2_31/S)+(hidden_layer14%*%W2_41/S))
  hidden_layer21 <- matrix(hidden_layer21, nrow  = N)
  
  hidden_layer22 <- relu(b2_2+(hidden_layer11%*%W2_12/S)+(hidden_layer12%*%W2_22/S)+(hidden_layer13%*%W2_32/S)+(hidden_layer14%*%W2_42/S))
  hidden_layer22 <- matrix(hidden_layer22, nrow  = N)
  
  hidden_layer23 <- relu(b2_3+(hidden_layer11%*%W2_13/S)+(hidden_layer12%*%W2_23/S)+(hidden_layer13%*%W2_33/S)+(hidden_layer14%*%W2_43/S))
  hidden_layer23 <- matrix(hidden_layer23, nrow  = N)
  
  hidden_layer24 <- relu(b2_4+(hidden_layer11%*%W2_14/S)+(hidden_layer12%*%W2_24/S)+(hidden_layer13%*%W2_34/S)+(hidden_layer14%*%W2_44/S))
  hidden_layer24 <- matrix(hidden_layer24, nrow  = N)
  
  
  # compute the loss: sofmax and regularization
  yhat=(b3_1+(hidden_layer21%*%W3_1/S)+(hidden_layer22%*%W3_2/S)+(hidden_layer23%*%W3_3/S)+(hidden_layer24%*%W3_4/S))
  
  
  
  return(yhat)  
}



fdnn.model <- fdnn4.4.es(X=xt, Y=Y,XV=xtv,YV=yv, S=30, step_size = 0.01, niteration = 5000)

t441=rmse(as.numeric(y),as.numeric(fdnn.pred4.4(X=xt, S=30, fdnn.model)))
t441


fdnn.pred.model <- fdnn.pred4.4(X=xtv, S=30, fdnn.model)
t4p41v=rmse(as.numeric(yv),as.numeric(fdnn.pred.model))

fdnn.pred.model <- fdnn.pred4.4(X=xtp, S=30, fdnn.model)
t4p41=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
t441
t4p41v
t4p41






####################################
#########FBNN22
####################################


#Here we are defining 2 hidden layers with 2 neurons each and using fBnn strategy

#vm and vn are the number of basis functions (bspline) used in the additional hidden layers.

fbnn2.2.es <- function(X, Y, XV, YV, S, vi, vj, vk,vm,vn, step_size = 0.5, niteration){
  
  xv=XV
  yv=YV
  xt=X
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  
  
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  vk1 <- seq(0, 1, by=1/(S-1))
  vk_v <- t(as.matrix(bSpline(vk1,df=vk)))
  
  vm1 <- seq(0, 1, by=1/(S-1))
  vm_v <- t(as.matrix(bSpline(vm1,df=vm)))
  
  vn1 <- seq(0, 1, by=1/(S-1))
  vn_v <- t(as.matrix(bSpline(vn1,df=vn)))
  
  
  # initialize parameters randomly
  W1_1 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_1 <- matrix(0, nrow = N, ncol = S)
  W1_2 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_2 <- matrix(0, nrow = N, ncol = S)
  
  W2_11 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_12 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_21 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_22 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  
  b2_1 <- matrix(0, nrow = N, ncol = S)
  b2_2 <- matrix(0, nrow = N, ncol = S)
  
  
  
  W3_1 <- 0.01 * as.vector(rnorm(vk))
  b3_1 <- matrix(0, nrow = N, ncol = 1)
  W3_2 <- 0.01 * as.vector(rnorm(vk))
  
  trainloss=c()
  testloss=c()
  
  W1_1list=list()  
  b1_1list=list()  
  W1_2list=list()  
  b1_2list=list()
  
  W2_11list=list()
  W2_12list=list()
  W2_21list=list()
  W2_22list=list()
  b2_1list=list()
  b2_2list=list()
  
  W3_1list=list()   
  W3_2list=list()  
  b3_1list=list()  
  
  yhatlist=list()
  
  
  
  
  
  # gradient descent loop to update weight and bias
  for (r in 0:niteration){
    # hidden layer, ReLU activation
    
    ai1_1=t(as.matrix(vi_v)%*%t(xt)/M)
    
    
    awv1_1.1=ai1_1%*%W1_1
    awv1_1=awv1_1.1%*%vj_v
    
    hidden_layer11 <- relu(b1_1+awv1_1)
    hidden_layer11<- matrix(hidden_layer11, nrow  = N)
    
    
    
    awv1_2.1=ai1_1%*%W1_2
    awv1_2=awv1_2.1%*%vj_v
    
    hidden_layer12 <- relu(b1_2+awv1_2)
    hidden_layer12<- matrix(hidden_layer12, nrow  = N)
    
    
    #layer 2
    ai2_11=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_11.1=ai2_11%*%W2_11
    awv2_11=awv2_11.1%*%vn_v
    
    ai2_21=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_21.1=ai2_21%*%W2_21
    awv2_21=awv2_21.1%*%vn_v
    
    hidden_layer21 <- relu(b2_1+awv2_11+awv2_21)
    hidden_layer21<- matrix(hidden_layer21, nrow  = N)
    
    
    
    ai2_12=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_12.1=ai2_12%*%W2_12
    awv2_12=awv2_12.1%*%vn_v
    
    ai2_22=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_22.1=ai2_22%*%W2_22
    awv2_22=awv2_22.1%*%vn_v
    
    hidden_layer22<- relu(b2_2+awv2_12+awv2_22)
    hidden_layer22<- matrix(hidden_layer22, nrow  = N)
    
    
    
    
    #last layer
    ai3_1=t(as.matrix(vk_v)%*%t(hidden_layer21)/S)
    
    awv3_1=ai3_1%*%W3_1
    
    
    ai3_2=t(as.matrix(vk_v)%*%t(hidden_layer22)/S)
    
    awv3_2=ai3_2%*%W3_2
    
    hidden_layer3 <- (b3_1+awv3_1+awv3_2)
    hidden_layer3<- matrix(hidden_layer3, nrow  = N)
    
    
    
    # compute the loss: sofmax and regularization
    yhat=hidden_layer3
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    
    
    ########validation
    Nv <- nrow(xtv)
    ai1_1test=t(as.matrix(vi_v)%*%t(xv)/M)
    
    
    awv1_1.1test=ai1_1test%*%W1_1
    awv1_1test=awv1_1.1test%*%vj_v
    
    hidden_layer11test <- relu(b1_1[1:Nv,]+awv1_1test)
    hidden_layer11test<- matrix(hidden_layer11test, nrow  = Nv)
    
    
    
    awv1_2.1test=ai1_1test%*%W1_2
    awv1_2test=awv1_2.1test%*%vj_v
    
    hidden_layer12test <- relu(b1_2[1:Nv,]+awv1_2test)
    hidden_layer12test<- matrix(hidden_layer12test, nrow  = Nv)
    
    
    #layer 2
    ai2_11test=t(as.matrix(vm_v)%*%t(hidden_layer11test)/S)
    
    awv2_11.1test=ai2_11test%*%W2_11
    awv2_11test=awv2_11.1test%*%vn_v
    
    ai2_21test=t(as.matrix(vm_v)%*%t(hidden_layer12test)/S)
    
    awv2_21.1test=ai2_21test%*%W2_21
    awv2_21test=awv2_21.1test%*%vn_v
    
    hidden_layer21test <- relu(b2_1[1:Nv,]+awv2_11test+awv2_21test)
    hidden_layer21test<- matrix(hidden_layer21test, nrow  = Nv)
    
    
    
    ai2_12test=t(as.matrix(vm_v)%*%t(hidden_layer11test)/S)
    
    awv2_12.1test=ai2_12test%*%W2_12
    awv2_12test=awv2_12.1test%*%vn_v
    
    ai2_22test=t(as.matrix(vm_v)%*%t(hidden_layer12test)/S)
    
    awv2_22.1test=ai2_22test%*%W2_22
    awv2_22test=awv2_22.1test%*%vn_v
    
    hidden_layer22test<- relu(b2_2[1:Nv,]+awv2_12test+awv2_22test)
    hidden_layer22test<- matrix(hidden_layer22test, nrow  = Nv)
    
    
    
    
    #last layer
    ai3_1test=t(as.matrix(vk_v)%*%t(hidden_layer21test)/S)
    
    awv3_1test=ai3_1test%*%W3_1
    
    
    ai3_2test=t(as.matrix(vk_v)%*%t(hidden_layer22test)/S)
    
    awv3_2test=ai3_2test%*%W3_2
    
    hidden_layer3test <- (b3_1[1:Nv]+awv3_1test+awv3_2test)
    hidden_layer3test<- matrix(hidden_layer3test, nrow  = Nv)
    
    
    
    # compute the loss: sofmax and regularization
    yhattest=hidden_layer3test
    data_losstest <- sum((yv-yhattest)^2)/N
    losstest <- data_losstest
    
    # check progress
    if (r%%1000 == 0 | r == niteration){
      print(paste("iteration", r,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    dW3_1=t(ai3_1)%*%dscores
    
    dW3_2=t(ai3_2)%*%dscores
    
    db3_1=colSums(dscores)
    
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    
    dW2_11.0=as.vector(W3_1)%*%vk_v
    dW2_11.1=dscores%*%dW2_11.0
    dW2_11.2=(as.matrix(relud(hidden_layer21)))
    dW2_11.3=dW2_11.1*dW2_11.2
    dW2_11.4=vn_v%*%t(dW2_11.3)
    dW2_11.5=dW2_11.4%*%(ai2_11)
    dW2_11=t(dW2_11.5) 
    
    
    db2_1=colSums(dW2_11.3)
    
    
    dW2_21.5=dW2_11.4%*%(ai2_21)
    dW2_21=t(dW2_21.5) 
    
    
    
    dW2_12.0=as.vector(W3_2)%*%vk_v
    dW2_12.1=dscores%*%dW2_12.0
    dW2_12.2=(as.matrix(relud(hidden_layer22)))
    dW2_12.3=dW2_12.1*dW2_12.2
    dW2_12.4=vn_v%*%t(dW2_12.3)
    dW2_12.5=dW2_12.4%*%(ai2_12)
    dW2_12=t(dW2_12.5) 
    
    
    db2_2=colSums(dW2_12.3)
    
    
    dW2_22.5=dW2_12.4%*%(ai2_22)
    dW2_22=t(dW2_22.5) 
    
    
    
    
    dW1_1.0=t(vm_v)%*%W2_11
    dW1_1.0=dW1_1.0%*%vn_v
    dW1_1.1=t(vm_v)%*%W2_12
    dW1_1.1=dW1_1.1%*%vn_v
    dW1_1.2=dW2_11.3%*%dW1_1.0+dW2_12.3%*%dW1_1.1
    dW1_1.3=(as.matrix(relud(hidden_layer11)))
    dW1_1.4=dW1_1.3*dW1_1.2
    dW1_1.5=vj_v%*%t(dW1_1.4)
    dW1_1.6=dW1_1.5%*%ai1_1
    dW1_1=t(dW1_1.6)
    
    db1_1=colSums(dW1_1.4)
    
    
    
    dW1_2.0=t(vm_v)%*%W2_21
    dW1_2.0=dW1_2.0%*%vn_v
    dW1_2.1=t(vm_v)%*%W2_22
    dW1_2.1=dW1_2.1%*%vn_v
    dW1_2.2=dW2_11.3%*%dW1_2.0+dW2_12.3%*%dW1_2.1
    dW1_2.3=(as.matrix(relud(hidden_layer12)))
    dW1_2.4=dW1_2.3*dW1_2.2
    dW1_2.5=vj_v%*%t(dW1_2.4)
    dW1_2.6=dW1_2.5%*%ai1_1
    dW1_2=t(dW1_2.6)
    
    db1_2=colSums(dW1_2.4)
    
    
    
    
    
    # update parameter
    W1_1 <- W1_1-step_size*dW1_1
    W1_2 <- W1_2-step_size*dW1_2
    b1_1 <- b1_1[1,]-step_size*db1_1
    b1_1=matrix(rep(b1_1,N),N,S, byrow = T)
    b1_2 <- b1_2[1,]-step_size*db1_2
    b1_2=matrix(rep(b1_2,N),N,S, byrow = T)
    
    W2_11 <- W2_11-step_size*dW2_11    
    W2_12 <- W2_12-step_size*dW2_12    
    W2_21 <- W2_21-step_size*dW2_21    
    W2_22 <- W2_22-step_size*dW2_22
    b2_1 <- b2_1[1,]-step_size*db2_1
    b2_1=matrix(rep(b2_1,N),N,S, byrow = T)
    b2_2 <- b2_2[1,]-step_size*db2_2
    b2_2=matrix(rep(b2_2,N),N,S, byrow = T)
    
    
    W3_1 <- W3_1-step_size*dW3_1
    W3_2 <- W3_2-step_size*dW3_2
    b3_1 <- b3_1[1]-step_size*db3_1
    b3_1=rep(b3_1,N)
    
    
    trainloss=c(trainloss,data_loss)
    testloss=c(testloss,data_losstest)
    
    W1_1list=list.append(W1_1list,W1_1)  
    b1_1list=list.append(b1_1list,b1_1)  
    W1_2list=list.append(W1_2list,W1_2)  
    b1_2list=list.append(b1_2list,b1_2)
    
    W2_11list=list.append(W2_11list,W2_11)
    W2_12list=list.append(W2_12list,W2_12)
    W2_21list=list.append(W2_21list,W2_21)
    W2_22list=list.append(W2_22list,W2_22)
    b2_1list=list.append(b2_1list,b2_1)
    b2_2list=list.append(b2_2list,b2_2)
    
    W3_1list=list.append(W3_1list,W3_1)   
    W3_2list=list.append(W3_2list,W3_2)  
    b3_1list=list.append(b3_1list,b3_1) 
    
    
    yhatlist=list.append(yhatlist,yhat)
    
    
  }
  kk=which.min(testloss)
  
  W1_1 <-W1_1list[[kk-1]]
  b1_1 <-b1_1list[[kk-1]]
  W1_2 <- W1_2list[[kk-1]]
  b1_2 <-b1_2list[[kk-1]]
  
  W2_11 <-W2_11list[[kk-1]]
  W2_12 <-W2_12list[[kk-1]]
  W2_21 <- W2_21list[[kk-1]]
  W2_22 <- W2_22list[[kk-1]]
  b2_1 <- b2_1list[[kk-1]]
  b2_2 <- b2_2list[[kk-1]]
  
  #second layer
  b3_1 <-b3_1list[[kk-1]]
  W3_1 <-W3_1list[[kk-1]]
  W3_2 <- W3_2list[[kk-1]]
  
  
  yhat=yhatlist[[kk-1]]
  loss=trainloss[kk]
  lossv=testloss[kk]
  
  return(list(W1_1, W1_2, b1_1, b1_2, 
              W2_11, W2_12, W2_21, W2_22, b2_1, b2_2, 
              W3_1, W3_2, b3_1,
              sqrt(loss),yhat,trainloss,testloss,which.min(trainloss),which.min(testloss),kk,sqrt(lossv)))
}



fbnn.pred2.2 <- function(X, S, vi, vj, vk,vm,vn, para = list()){
  
  W1_1 <-para[[1]]
  W1_2 <-para[[2]]
  
  
  b1_1 <-para[[3]]
  b1_1=b1_1[(1:length(X[,1])),]
  b1_2 <-para[[4]]
  b1_2=b1_2[(1:length(X[,1])),]
  
  
  #second layer
  
  W2_11 <-para[[5]]
  W2_12 <-para[[6]]
  W2_21 <-para[[7]]
  W2_22 <-para[[8]]
  
  
  b2_1 <-para[[9]]
  b2_1=b2_1[(1:length(X[,1])),]
  
  b2_2 <-para[[10]]
  b2_2=b2_2[(1:length(X[,1])),]
  
  
  
  W3_1 <-para[[11]]
  W3_2 <-para[[12]]
  
  b3_1 <-para[[13]]
  b3_1=b3_1[(1:length(X[,1]))]
  
  
  
  
  
  
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  
  
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  vk1 <- seq(0, 1, by=1/(S-1))
  vk_v <- t(as.matrix(bSpline(vk1,df=vk)))
  
  vm1 <- seq(0, 1, by=1/(S-1))
  vm_v <- t(as.matrix(bSpline(vm1,df=vm)))
  
  vn1 <- seq(0, 1, by=1/(S-1))
  vn_v <- t(as.matrix(bSpline(vn1,df=vn)))
  
  
  
  ai1_1=t(as.matrix(vi_v)%*%t(xt)/M)
  
  
  awv1_1.1=ai1_1%*%W1_1
  awv1_1=awv1_1.1%*%vj_v
  
  hidden_layer11 <- relu(b1_1+awv1_1)
  hidden_layer11<- matrix(hidden_layer11, nrow  = N)
  
  
  
  awv1_2.1=ai1_1%*%W1_2
  awv1_2=awv1_2.1%*%vj_v
  
  hidden_layer12 <- relu(b1_2+awv1_2)
  hidden_layer12<- matrix(hidden_layer12, nrow  = N)
  
  
  #layer 2
  ai2_11=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_11.1=ai2_11%*%W2_11
  awv2_11=awv2_11.1%*%vn_v
  
  ai2_21=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_21.1=ai2_21%*%W2_21
  awv2_21=awv2_21.1%*%vn_v
  
  hidden_layer21 <- relu(b2_1+awv2_11+awv2_21)
  hidden_layer21<- matrix(hidden_layer21, nrow  = N)
  
  
  
  ai2_12=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_12.1=ai2_12%*%W2_12
  awv2_12=awv2_12.1%*%vn_v
  
  ai2_22=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_22.1=ai2_22%*%W2_22
  awv2_22=awv2_22.1%*%vn_v
  
  hidden_layer22<- relu(b2_2+awv2_12+awv2_22)
  hidden_layer22<- matrix(hidden_layer22, nrow  = N)
  
  
  
  
  #last layer
  ai3_1=t(as.matrix(vk_v)%*%t(hidden_layer21)/S)
  
  awv3_1=ai3_1%*%W3_1
  
  
  ai3_2=t(as.matrix(vk_v)%*%t(hidden_layer22)/S)
  
  awv3_2=ai3_2%*%W3_2
  
  hidden_layer3 <- (b3_1+awv3_1+awv3_2)
  hidden_layer3<- matrix(hidden_layer3, nrow  = N)
  
  
  
  # compute the loss: sofmax and regularization
  yhat=hidden_layer3
  
  return(yhat)  
}



fbnn.model <- fbnn2.2.es(X=xt, Y=Y,XV=xtv,YV=yv, S=30, vi=5, vj=5, vk=5,vm=5,vn=5, step_size=.001, niteration = 5000)



t4_i2=rmse(as.numeric(y),as.numeric(fbnn.pred2.2(X=xt, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)))
t4_i2

fbnn.pred.model <- fbnn.pred2.2(X=xtv, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)
t4p_i2v=rmse(as.numeric(yv),as.numeric(fbnn.pred.model))

fbnn.pred.model <- fbnn.pred2.2(X=xtp, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)
t4p_i2=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
t4_i2
t4p_i2





####################################
#########FBNN44
####################################


#Here we are defining 2 hidden layers with 4 neurons each and using fBnn strategy

#vm and vn are the number of basis functions (bspline) used in the additional hidden layers.

fbnn4.4.es <- function(X, Y,XV,YV, S, vi, vj, vk,vm,vn, step_size = 0.5, niteration){
  xv=XV
  yv=YV
  xt=X
  xt=X
  # get dim of input
  N <- nrow(xt) # number of examples
  M <- ncol(xt) # number of timepoints for xt
  S <- S # number of timepoints on s
  
  
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  vk1 <- seq(0, 1, by=1/(S-1))
  vk_v <- t(as.matrix(bSpline(vk1,df=vk)))
  
  vm1 <- seq(0, 1, by=1/(S-1))
  vm_v <- t(as.matrix(bSpline(vm1,df=vm)))
  
  vn1 <- seq(0, 1, by=1/(S-1))
  vn_v <- t(as.matrix(bSpline(vn1,df=vn)))
  
  
  # initialize parameters randomly
  W1_1 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_1 <- matrix(0, nrow = N, ncol = S)
  W1_2 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_2 <- matrix(0, nrow = N, ncol = S)
  W1_3 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_3 <- matrix(0, nrow = N, ncol = S)
  W1_4 <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b1_4 <- matrix(0, nrow = N, ncol = S)
  
  
  W2_11 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_12 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_13 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_14 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_21 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_22 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_23 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_24 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_31 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_32 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_33 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_34 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_41 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_42 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)  
  W2_43 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  W2_44 <- 0.01 * matrix(rnorm(vm*vn), nrow = vm)
  
  
  b2_1 <- matrix(0, nrow = N, ncol = S)
  b2_2 <- matrix(0, nrow = N, ncol = S)  
  b2_3 <- matrix(0, nrow = N, ncol = S)
  b2_4 <- matrix(0, nrow = N, ncol = S)
  
  
  
  W3_1 <- 0.01 * as.vector(rnorm(vk))
  b3_1 <- matrix(0, nrow = N, ncol = 1)
  W3_2 <- 0.01 * as.vector(rnorm(vk))
  W3_3 <- 0.01 * as.vector(rnorm(vk))
  W3_4 <- 0.01 * as.vector(rnorm(vk))
  
  
  
  
  trainloss=c()
  testloss=c()
  
  
  W1_1list=list()  
  b1_1list=list()  
  W1_2list=list()  
  b1_2list=list()
  
  W1_3list=list()  
  b1_3list=list()  
  W1_4list=list()  
  b1_4list=list()
  
  W2_11list=list()
  W2_21list=list()
  W2_31list=list()
  W2_41list=list()
  
  W2_12list=list()
  W2_22list=list()
  W2_32list=list()
  W2_42list=list()
  
  W2_13list=list()
  W2_23list=list()
  W2_33list=list()
  W2_43list=list()
  
  W2_14list=list()
  W2_24list=list()
  W2_34list=list()
  W2_44list=list()
  
  b2_1list=list()
  b2_2list=list()
  b2_3list=list()
  b2_4list=list()
  
  
  
  W3_1list=list()
  W3_2list=list()
  W3_3list=list()
  W3_4list=list()
  
  b3_1list=list()
  
  yhatlist=list()
  
  
  # gradient descent loop to update weight and bias
  for (r in 0:niteration){
    #hidden layer, ReLU activation
    
    ai1_1=t(as.matrix(vi_v)%*%t(xt)/M)
    
    
    awv1_1.1=ai1_1%*%W1_1
    awv1_1=awv1_1.1%*%vj_v
    
    hidden_layer11 <- relu(b1_1+awv1_1)
    hidden_layer11<- matrix(hidden_layer11, nrow  = N)
    
    
    
    awv1_2.1=ai1_1%*%W1_2
    awv1_2=awv1_2.1%*%vj_v
    
    hidden_layer12 <- relu(b1_2+awv1_2)
    hidden_layer12<- matrix(hidden_layer12, nrow  = N)
    
    awv1_3.1=ai1_1%*%W1_3
    awv1_3=awv1_3.1%*%vj_v
    
    hidden_layer13 <- relu(b1_3+awv1_3)
    hidden_layer13<- matrix(hidden_layer13, nrow  = N)
    
    awv1_4.1=ai1_1%*%W1_4
    awv1_4=awv1_4.1%*%vj_v
    
    hidden_layer14 <- relu(b1_4+awv1_4)
    hidden_layer14<- matrix(hidden_layer14, nrow  = N)
    
    
    #layer 2
    ai2_11=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_11.1=ai2_11%*%W2_11
    awv2_11=awv2_11.1%*%vn_v
    
    ai2_21=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_21.1=ai2_21%*%W2_21
    awv2_21=awv2_21.1%*%vn_v
    
    ai2_31=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_31.1=ai2_31%*%W2_31
    awv2_31=awv2_31.1%*%vn_v
    
    ai2_41=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_41.1=ai2_41%*%W2_41
    awv2_41=awv2_41.1%*%vn_v
    
    
    hidden_layer21 <- relu(b2_1+awv2_11+awv2_21+awv2_31+awv2_41)
    hidden_layer21<- matrix(hidden_layer21, nrow  = N)
    
    
    
    ai2_12=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_12.1=ai2_12%*%W2_12
    awv2_12=awv2_12.1%*%vn_v
    
    ai2_22=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_22.1=ai2_22%*%W2_22
    awv2_22=awv2_22.1%*%vn_v
    
    ai2_32=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_32.1=ai2_32%*%W2_32
    awv2_32=awv2_32.1%*%vn_v
    
    ai2_42=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_42.1=ai2_42%*%W2_42
    awv2_42=awv2_42.1%*%vn_v
    
    
    hidden_layer22 <- relu(b2_2+awv2_12+awv2_22+awv2_32+awv2_42)
    hidden_layer22<- matrix(hidden_layer22, nrow  = N)
    
    
    
    ai2_13=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_13.1=ai2_13%*%W2_13
    awv2_13=awv2_13.1%*%vn_v
    
    ai2_23=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_23.1=ai2_23%*%W2_23
    awv2_23=awv2_23.1%*%vn_v
    
    ai2_33=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_33.1=ai2_33%*%W2_33
    awv2_33=awv2_33.1%*%vn_v
    
    ai2_43=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_43.1=ai2_43%*%W2_43
    awv2_43=awv2_43.1%*%vn_v
    
    
    hidden_layer23 <- relu(b2_3+awv2_13+awv2_23+awv2_33+awv2_43)
    hidden_layer23<- matrix(hidden_layer23, nrow  = N)
    
    
    ai2_14=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
    
    awv2_14.1=ai2_14%*%W2_14
    awv2_14=awv2_14.1%*%vn_v
    
    ai2_24=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
    
    awv2_24.1=ai2_24%*%W2_24
    awv2_24=awv2_24.1%*%vn_v
    
    ai2_34=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
    
    awv2_34.1=ai2_34%*%W2_34
    awv2_34=awv2_34.1%*%vn_v
    
    ai2_44=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
    
    awv2_44.1=ai2_44%*%W2_44
    awv2_44=awv2_44.1%*%vn_v
    
    
    hidden_layer24 <- relu(b2_4+awv2_14+awv2_24+awv2_34+awv2_44)
    hidden_layer24<- matrix(hidden_layer24, nrow  = N)
    
    #last layer
    ai3_1=t(as.matrix(vk_v)%*%t(hidden_layer21)/S)
    
    awv3_1=ai3_1%*%W3_1
    
    
    ai3_2=t(as.matrix(vk_v)%*%t(hidden_layer22)/S)
    
    awv3_2=ai3_2%*%W3_2
    
    ai3_3=t(as.matrix(vk_v)%*%t(hidden_layer23)/S)
    
    awv3_3=ai3_3%*%W3_3
    
    
    ai3_4=t(as.matrix(vk_v)%*%t(hidden_layer24)/S)
    
    awv3_4=ai3_4%*%W3_4
    
    
    
    hidden_layer3 <- (b3_1+awv3_1+awv3_2+awv3_3+awv3_4)
    hidden_layer3<- matrix(hidden_layer3, nrow  = N)
    
    
    
    # compute the loss: sofmax and regularization
    yhat=hidden_layer3
    data_loss <- sum((Y-yhat)^2)/N
    loss <- data_loss
    
    
    
    
    #validation
    Nv <- nrow(xtv)
    
    ai1_1test=t(as.matrix(vi_v)%*%t(xv)/M)
    
    
    awv1_1.1test=ai1_1test%*%W1_1
    awv1_1test=awv1_1.1test%*%vj_v
    
    hidden_layer11test <- relu(b1_1[1:Nv,]+awv1_1test)
    hidden_layer11test<- matrix(hidden_layer11test, nrow  = Nv)
    
    
    
    awv1_2.1test=ai1_1test%*%W1_2
    awv1_2test=awv1_2.1test%*%vj_v
    
    hidden_layer12test <- relu(b1_2[1:Nv,]+awv1_2test)
    hidden_layer12test<- matrix(hidden_layer12test, nrow  = Nv)
    
    awv1_3.1test=ai1_1test%*%W1_3
    awv1_3test=awv1_3.1test%*%vj_v
    
    hidden_layer13test <- relu(b1_3[1:Nv,]+awv1_3test)
    hidden_layer13test<- matrix(hidden_layer13test, nrow  = Nv)
    
    awv1_4.1test=ai1_1test%*%W1_4
    awv1_4test=awv1_4.1test%*%vj_v
    
    hidden_layer14test <- relu(b1_4[1:Nv,]+awv1_4test)
    hidden_layer14test<- matrix(hidden_layer14test, nrow  = Nv)
    
    
    #layer 2
    ai2_11test=t(as.matrix(vm_v)%*%t(hidden_layer11test)/S)
    
    awv2_11.1test=ai2_11test%*%W2_11
    awv2_11test=awv2_11.1test%*%vn_v
    
    ai2_21test=t(as.matrix(vm_v)%*%t(hidden_layer12test)/S)
    
    awv2_21.1test=ai2_21test%*%W2_21
    awv2_21test=awv2_21.1test%*%vn_v
    
    ai2_31test=t(as.matrix(vm_v)%*%t(hidden_layer13test)/S)
    
    awv2_31.1test=ai2_31test%*%W2_31
    awv2_31test=awv2_31.1test%*%vn_v
    
    ai2_41test=t(as.matrix(vm_v)%*%t(hidden_layer14test)/S)
    
    awv2_41.1test=ai2_41test%*%W2_41
    awv2_41test=awv2_41.1test%*%vn_v
    
    
    hidden_layer21test <- relu(b2_1[1:Nv,]+awv2_11test+awv2_21test+awv2_31test+awv2_41test)
    hidden_layer21test<- matrix(hidden_layer21test, nrow  = Nv)
    
    
    
    ai2_12test=t(as.matrix(vm_v)%*%t(hidden_layer11test)/S)
    
    awv2_12.1test=ai2_12test%*%W2_12
    awv2_12test=awv2_12.1test%*%vn_v
    
    ai2_22test=t(as.matrix(vm_v)%*%t(hidden_layer12test)/S)
    
    awv2_22.1test=ai2_22test%*%W2_22
    awv2_22test=awv2_22.1test%*%vn_v
    
    ai2_32test=t(as.matrix(vm_v)%*%t(hidden_layer13test)/S)
    
    awv2_32.1test=ai2_32test%*%W2_32
    awv2_32test=awv2_32.1test%*%vn_v
    
    ai2_42test=t(as.matrix(vm_v)%*%t(hidden_layer14test)/S)
    
    awv2_42.1test=ai2_42test%*%W2_42
    awv2_42test=awv2_42.1test%*%vn_v
    
    
    hidden_layer22test <- relu(b2_2[1:Nv,]+awv2_12test+awv2_22test+awv2_32test+awv2_42test)
    hidden_layer22test<- matrix(hidden_layer22test, nrow  = Nv)
    
    
    
    ai2_13test=t(as.matrix(vm_v)%*%t(hidden_layer11test)/S)
    
    awv2_13.1test=ai2_13test%*%W2_13
    awv2_13test=awv2_13.1test%*%vn_v
    
    ai2_23test=t(as.matrix(vm_v)%*%t(hidden_layer12test)/S)
    
    awv2_23.1test=ai2_23test%*%W2_23
    awv2_23test=awv2_23.1test%*%vn_v
    
    ai2_33test=t(as.matrix(vm_v)%*%t(hidden_layer13test)/S)
    
    awv2_33.1test=ai2_33test%*%W2_33
    awv2_33test=awv2_33.1test%*%vn_v
    
    ai2_43test=t(as.matrix(vm_v)%*%t(hidden_layer14test)/S)
    
    awv2_43.1test=ai2_43test%*%W2_43
    awv2_43test=awv2_43.1test%*%vn_v
    
    
    hidden_layer23test <- relu(b2_3[1:Nv,]+awv2_13test+awv2_23test+awv2_33test+awv2_43test)
    hidden_layer23test<- matrix(hidden_layer23test, nrow  = Nv)
    
    
    ai2_14test=t(as.matrix(vm_v)%*%t(hidden_layer11test)/S)
    
    awv2_14.1test=ai2_14test%*%W2_14
    awv2_14test=awv2_14.1test%*%vn_v
    
    ai2_24test=t(as.matrix(vm_v)%*%t(hidden_layer12test)/S)
    
    awv2_24.1test=ai2_24test%*%W2_24
    awv2_24test=awv2_24.1test%*%vn_v
    
    ai2_34test=t(as.matrix(vm_v)%*%t(hidden_layer13test)/S)
    
    awv2_34.1test=ai2_34test%*%W2_34
    awv2_34test=awv2_34.1test%*%vn_v
    
    ai2_44test=t(as.matrix(vm_v)%*%t(hidden_layer14test)/S)
    
    awv2_44.1test=ai2_44test%*%W2_44
    awv2_44test=awv2_44.1test%*%vn_v
    
    
    hidden_layer24test <- relu(b2_4[1:Nv,]+awv2_14test+awv2_24test+awv2_34test+awv2_44test)
    hidden_layer24test<- matrix(hidden_layer24test, nrow  = Nv)
    
    #last layer
    ai3_1test=t(as.matrix(vk_v)%*%t(hidden_layer21test)/S)
    
    awv3_1test=ai3_1test%*%W3_1
    
    
    ai3_2test=t(as.matrix(vk_v)%*%t(hidden_layer22test)/S)
    
    awv3_2test=ai3_2test%*%W3_2
    
    ai3_3test=t(as.matrix(vk_v)%*%t(hidden_layer23test)/S)
    
    awv3_3test=ai3_3test%*%W3_3
    
    
    ai3_4test=t(as.matrix(vk_v)%*%t(hidden_layer24test)/S)
    
    awv3_4test=ai3_4test%*%W3_4
    
    
    
    hidden_layer3test <- (b3_1[1:Nv]+awv3_1test+awv3_2test+awv3_3test+awv3_4test)
    hidden_layer3test<- matrix(hidden_layer3test, nrow  = Nv)
    
    
    
    # compute the loss: sofmax and regularization
    yhattest=hidden_layer3test
    data_losstest <- sum((yv-yhattest)^2)/N
    losstest <- data_losstest
    
    # check progress
    if (r%%1000 == 0 | r == niteration){
      print(paste("iteration", r,': loss', loss))}
    
    # compute the gradient on scores
    dscores= -2*(Y-yhat)/N
    
    # backpropate the gradient to the parameters
    dW3_1=t(ai3_1)%*%dscores
    
    dW3_2=t(ai3_2)%*%dscores    
    dW3_3=t(ai3_3)%*%dscores    
    dW3_4=t(ai3_4)%*%dscores
    
    db3_1=colSums(dscores)
    
    ##########################################################################################correct form here    
    # next backprop into hidden layer
    
    dW2_11.0=as.vector(W3_1)%*%vk_v
    dW2_11.1=dscores%*%dW2_11.0
    dW2_11.2=(as.matrix(relud(hidden_layer21)))
    dW2_11.3=dW2_11.1*dW2_11.2
    dW2_11.4=vn_v%*%t(dW2_11.3)
    dW2_11.5=dW2_11.4%*%(ai2_11)
    dW2_11=t(dW2_11.5) 
    
    
    db2_1=colSums(dW2_11.3)
    
    
    dW2_21.5=dW2_11.4%*%(ai2_21)
    dW2_21=t(dW2_21.5) 
    
    dW2_31.5=dW2_11.4%*%(ai2_31)
    dW2_31=t(dW2_31.5) 
    
    dW2_41.5=dW2_11.4%*%(ai2_41)
    dW2_41=t(dW2_41.5) 
    
    
    
    dW2_12.0=as.vector(W3_2)%*%vk_v
    dW2_12.1=dscores%*%dW2_12.0
    dW2_12.2=(as.matrix(relud(hidden_layer22)))
    dW2_12.3=dW2_12.1*dW2_12.2
    dW2_12.4=vn_v%*%t(dW2_12.3)
    dW2_12.5=dW2_12.4%*%(ai2_12)
    dW2_12=t(dW2_12.5) 
    
    
    db2_2=colSums(dW2_12.3)
    
    
    dW2_22.5=dW2_12.4%*%(ai2_22)
    dW2_22=t(dW2_22.5) 
    
    dW2_32.5=dW2_12.4%*%(ai2_23)
    dW2_32=t(dW2_32.5) 
    
    dW2_42.5=dW2_12.4%*%(ai2_24)
    dW2_42=t(dW2_42.5) 
    
    
    
    
    
    
    dW2_13.0=as.vector(W3_3)%*%vk_v
    dW2_13.1=dscores%*%dW2_13.0
    dW2_13.2=(as.matrix(relud(hidden_layer23)))
    dW2_13.3=dW2_13.1*dW2_13.2
    dW2_13.4=vn_v%*%t(dW2_13.3)
    dW2_13.5=dW2_13.4%*%(ai2_13)
    dW2_13=t(dW2_13.5) 
    
    
    db2_3=colSums(dW2_13.3)
    
    
    dW2_23.5=dW2_13.4%*%(ai2_23)
    dW2_23=t(dW2_23.5) 
    
    dW2_33.5=dW2_13.4%*%(ai2_33)
    dW2_33=t(dW2_33.5) 
    
    dW2_43.5=dW2_13.4%*%(ai2_43)
    dW2_43=t(dW2_43.5) 
    
    
    
    dW2_14.0=as.vector(W3_4)%*%vk_v
    dW2_14.1=dscores%*%dW2_14.0
    dW2_14.2=(as.matrix(relud(hidden_layer24)))
    dW2_14.3=dW2_14.1*dW2_14.2
    dW2_14.4=vn_v%*%t(dW2_14.3)
    dW2_14.5=dW2_14.4%*%(ai2_14)
    dW2_14=t(dW2_14.5) 
    
    
    db2_4=colSums(dW2_14.3)
    
    
    dW2_24.5=dW2_14.4%*%(ai2_24)
    dW2_24=t(dW2_24.5)
    
    dW2_34.5=dW2_14.4%*%(ai2_34)
    dW2_34=t(dW2_34.5) 
    
    dW2_44.5=dW2_14.4%*%(ai2_44)
    dW2_44=t(dW2_44.5) 
    
    
    
    
    
    dW1_1.0=t(vm_v)%*%W2_11
    dW1_1.0=dW1_1.0%*%vn_v
    dW1_1.1=t(vm_v)%*%W2_12
    dW1_1.1=dW1_1.1%*%vn_v
    dW1_1.11=t(vm_v)%*%W2_13
    dW1_1.11=dW1_1.11%*%vn_v
    dW1_1.111=t(vm_v)%*%W2_14
    dW1_1.111=dW1_1.111%*%vn_v
    dW1_1.2=dW2_11.3%*%dW1_1.0+dW2_12.3%*%dW1_1.1+dW2_13.3%*%dW1_1.11+dW2_14.3%*%dW1_1.111
    dW1_1.3=(as.matrix(relud(hidden_layer11)))
    dW1_1.4=dW1_1.3*dW1_1.2
    dW1_1.5=vj_v%*%t(dW1_1.4)
    dW1_1.6=dW1_1.5%*%ai1_1
    dW1_1=t(dW1_1.6)
    
    db1_1=colSums(dW1_1.4)
    
    
    
    dW1_2.0=t(vm_v)%*%W2_21
    dW1_2.0=dW1_2.0%*%vn_v
    dW1_2.1=t(vm_v)%*%W2_22
    dW1_2.1=dW1_2.1%*%vn_v
    dW1_2.11=t(vm_v)%*%W2_23
    dW1_2.11=dW1_2.11%*%vn_v
    dW1_2.111=t(vm_v)%*%W2_24
    dW1_2.111=dW1_2.111%*%vn_v
    dW1_2.2=dW2_11.3%*%dW1_2.0+dW2_12.3%*%dW1_2.1+dW2_13.3%*%dW1_2.11+dW2_14.3%*%dW1_2.111
    dW1_2.3=(as.matrix(relud(hidden_layer12)))
    dW1_2.4=dW1_2.3*dW1_2.2
    dW1_2.5=vj_v%*%t(dW1_2.4)
    dW1_2.6=dW1_2.5%*%ai1_1
    dW1_2=t(dW1_2.6)
    
    db1_2=colSums(dW1_2.4)
    
    
    
    dW1_3.0=t(vm_v)%*%W2_31
    dW1_3.0=dW1_3.0%*%vn_v
    dW1_3.1=t(vm_v)%*%W2_32
    dW1_3.1=dW1_3.1%*%vn_v
    dW1_3.11=t(vm_v)%*%W2_33
    dW1_3.11=dW1_3.11%*%vn_v
    dW1_3.111=t(vm_v)%*%W2_34
    dW1_3.111=dW1_3.111%*%vn_v
    dW1_3.2=dW2_11.3%*%dW1_3.0+dW2_12.3%*%dW1_3.1+dW2_13.3%*%dW1_3.11+dW2_14.3%*%dW1_3.111
    dW1_3.3=(as.matrix(relud(hidden_layer13)))
    dW1_3.4=dW1_3.3*dW1_3.2
    dW1_3.5=vj_v%*%t(dW1_3.4)
    dW1_3.6=dW1_3.5%*%ai1_1
    dW1_3=t(dW1_3.6)
    
    db1_3=colSums(dW1_3.4)
    
    
    
    dW1_4.0=t(vm_v)%*%W2_41
    dW1_4.0=dW1_4.0%*%vn_v
    dW1_4.1=t(vm_v)%*%W2_42
    dW1_4.1=dW1_4.1%*%vn_v
    dW1_4.11=t(vm_v)%*%W2_43
    dW1_4.11=dW1_4.11%*%vn_v
    dW1_4.111=t(vm_v)%*%W2_44
    dW1_4.111=dW1_4.111%*%vn_v
    dW1_4.2=dW2_11.3%*%dW1_4.0+dW2_12.3%*%dW1_4.1+dW2_13.3%*%dW1_4.11+dW2_14.3%*%dW1_4.111
    dW1_4.3=(as.matrix(relud(hidden_layer14)))
    dW1_4.4=dW1_4.3*dW1_4.2
    dW1_4.5=vj_v%*%t(dW1_4.4)
    dW1_4.6=dW1_4.5%*%ai1_1
    dW1_4=t(dW1_4.6)
    
    db1_4=colSums(dW1_4.4)
    
    
    
    
    # update parameter
    W1_1 <- W1_1-step_size*dW1_1
    W1_2 <- W1_2-step_size*dW1_2
    b1_1 <- b1_1[1,]-step_size*db1_1
    b1_1=matrix(rep(b1_1,N),N,S, byrow = T)
    b1_2 <- b1_2[1,]-step_size*db1_2
    b1_2=matrix(rep(b1_2,N),N,S, byrow = T)
    
    W1_3 <- W1_3-step_size*dW1_3
    W1_4 <- W1_4-step_size*dW1_4
    b1_3 <- b1_3[1,]-step_size*db1_3
    b1_3=matrix(rep(b1_3,N),N,S, byrow = T)
    b1_4 <- b1_4[1,]-step_size*db1_4
    b1_4=matrix(rep(b1_4,N),N,S, byrow = T)
    
    W2_11 <- W2_11-step_size*dW2_11    
    W2_12 <- W2_12-step_size*dW2_12     
    W2_13 <- W2_13-step_size*dW2_13    
    W2_14 <- W2_14-step_size*dW2_14
    
    
    W2_21 <- W2_21-step_size*dW2_21    
    W2_22 <- W2_22-step_size*dW2_22     
    W2_23 <- W2_23-step_size*dW2_23    
    W2_24 <- W2_24-step_size*dW2_24  
    
    W2_31 <- W2_31-step_size*dW2_31    
    W2_32 <- W2_32-step_size*dW2_32     
    W2_33 <- W2_33-step_size*dW2_33    
    W2_34 <- W2_34-step_size*dW2_34
    
    
    W2_41 <- W2_41-step_size*dW2_41    
    W2_42 <- W2_42-step_size*dW2_42     
    W2_43 <- W2_43-step_size*dW2_43    
    W2_44 <- W2_44-step_size*dW2_44 
    
    b2_1 <- b2_1[1,]-step_size*db2_1
    b2_1=matrix(rep(b2_1,N),N,S, byrow = T)
    b2_2 <- b2_2[1,]-step_size*db2_2
    b2_2=matrix(rep(b2_2,N),N,S, byrow = T)
    
    b2_3 <- b2_3[1,]-step_size*db2_3
    b2_3=matrix(rep(b2_3,N),N,S, byrow = T)
    b2_4 <- b2_4[1,]-step_size*db2_4
    b2_4=matrix(rep(b2_4,N),N,S, byrow = T)
    
    
    W3_1 <- W3_1-step_size*dW3_1
    W3_2 <- W3_2-step_size*dW3_2    
    W3_3 <- W3_3-step_size*dW3_3
    W3_4 <- W3_4-step_size*dW3_4
    b3_1 <- b3_1[1]-step_size*db3_1
    b3_1=rep(b3_1,N)
    
    
    trainloss=c(trainloss,data_loss)
    testloss=c(testloss,data_losstest)
    
    
    
    W1_1list=list.append(W1_1list,W1_1)  
    b1_1list=list.append(b1_1list,b1_1)  
    W1_2list=list.append(W1_2list,W1_2)  
    b1_2list=list.append(b1_2list,b1_2)
    
    W1_3list=list.append(W1_3list,W1_3)  
    b1_3list=list.append(b1_3list,b1_3)  
    W1_4list=list.append(W1_4list,W1_4)  
    b1_4list=list.append(b1_4list,b1_4)
    
    W2_11list=list.append(W2_11list,W2_11)
    W2_21list=list.append(W2_21list,W2_21)
    W2_31list=list.append(W2_31list,W2_31)
    W2_41list=list.append(W2_41list,W2_41)
    
    W2_12list=list.append(W2_12list,W2_12)
    W2_22list=list.append(W2_22list,W2_22)
    W2_32list=list.append(W2_32list,W2_32)
    W2_42list=list.append(W2_42list,W2_42)
    
    W2_13list=list.append(W2_13list,W2_13)
    W2_23list=list.append(W2_23list,W2_23)
    W2_33list=list.append(W2_33list,W2_33)
    W2_43list=list.append(W2_43list,W2_43)
    
    W2_14list=list.append(W2_14list,W2_14)
    W2_24list=list.append(W2_24list,W2_24)
    W2_34list=list.append(W2_34list,W2_34)
    W2_44list=list.append(W2_44list,W2_44)
    
    b2_1list=list.append(b2_1list,b2_1)
    b2_2list=list.append(b2_2list,b2_2)
    b2_3list=list.append(b2_3list,b2_3)
    b2_4list=list.append(b2_4list,b2_4)
    
    
    
    W3_1list=list.append(W3_1list,W3_1)
    W3_2list=list.append(W3_2list,W3_2)
    W3_3list=list.append(W3_3list,W3_3)
    W3_4list=list.append(W3_4list,W3_4)
    
    b3_1list=list.append(b3_1list,b3_1)
    
    
    yhatlist=list.append(yhatlist,yhat)
    
    
    
    
  }
  kk=which.min(testloss)
  
  W1_1 <-W1_1list[[kk-1]]
  b1_1 <-b1_1list[[kk-1]]
  W1_2 <- W1_2list[[kk-1]]
  b1_2 <-b1_2list[[kk-1]]
  
  
  W1_3 <-W1_3list[[kk-1]]
  b1_3 <-b1_3list[[kk-1]]
  W1_4 <- W1_4list[[kk-1]]
  b1_4 <-b1_4list[[kk-1]]
  
  W2_11 <-W2_11list[[kk-1]]
  W2_21 <-W2_21list[[kk-1]]
  W2_31 <-W2_31list[[kk-1]]
  W2_41 <-W2_41list[[kk-1]]
  
  W2_12 <-W2_12list[[kk-1]]
  W2_22 <-W2_22list[[kk-1]]
  W2_32 <-W2_32list[[kk-1]]
  W2_42 <-W2_42list[[kk-1]]
  
  W2_13 <-W2_13list[[kk-1]]
  W2_23 <-W2_23list[[kk-1]]
  W2_33 <-W2_33list[[kk-1]]
  W2_43 <-W2_43list[[kk-1]]
  
  W2_14 <-W2_14list[[kk-1]]
  W2_24 <-W2_24list[[kk-1]]
  W2_34 <-W2_34list[[kk-1]]
  W2_44 <-W2_44list[[kk-1]]
  
  b2_1 <-b2_1list[[kk-1]]
  b2_2 <-b2_2list[[kk-1]]
  b2_3 <-b2_3list[[kk-1]]
  b2_4 <-b2_4list[[kk-1]]
  #second layer
  
  W3_1 <- W3_1list[[kk-1]]
  W3_2 <- W3_2list[[kk-1]]
  W3_3 <- W3_3list[[kk-1]]
  W3_4 <- W3_4list[[kk-1]]
  
  b3_1 <-b3_1list[[kk-1]]
  
  
  yhat=yhatlist[[kk-1]]
  loss=trainloss[kk]
  lossv=testloss[kk]
  
  return(list(W1_1, W1_2, W1_3, W1_4, b1_1, b1_2, b1_3, b1_4, 
              W2_11, W2_12, W2_13, W2_14, W2_21, W2_22, W2_23, W2_24, W2_31, W2_32, W2_33, W2_34, W2_41, W2_42, W2_43, W2_44,
              b2_1, b2_2, b2_3, b2_4, 
              W3_1, W3_2, W3_3, W3_4, b3_1,
              sqrt(loss),yhat,trainloss,testloss,which.min(trainloss),which.min(testloss),kk,sqrt(lossv)))
}

fbnn.pred4.4 <- function(X, S, vi, vj, vk,vm,vn, para = list()){
  
  
  
  W1_1 <-para[[1]]
  W1_2 <-para[[2]]  
  W1_3 <-para[[3]]
  W1_4 <-para[[4]]
  
  
  b1_1 <-para[[5]]
  b1_2 <-para[[6]]  
  b1_3 <-para[[7]]
  b1_4 <-para[[8]]
  
  b1_1=b1_1[(1:length(X[,1])),]
  b1_2=b1_2[(1:length(X[,1])),]
  b1_3=b1_3[(1:length(X[,1])),]
  b1_4=b1_4[(1:length(X[,1])),]
  
  
  #second layer
  
  W2_11 <-para[[9]]
  W2_12 <-para[[10]]
  W2_13 <-para[[11]]
  W2_14 <-para[[12]]  
  W2_21 <-para[[13]]
  W2_22 <-para[[14]]
  W2_23 <-para[[15]]
  W2_24 <-para[[16]]  
  W2_31 <-para[[17]]
  W2_32 <-para[[18]]
  W2_33 <-para[[19]]
  W2_34 <-para[[20]]  
  W2_41 <-para[[21]]
  W2_42 <-para[[22]]
  W2_43 <-para[[23]]
  W2_44 <-para[[24]]
  
  
  b2_1 <-para[[25]]
  b2_2 <-para[[26]]  
  b2_3 <-para[[27]]
  b2_4 <-para[[28]]
  
  b2_1=b2_1[(1:length(X[,1])),]
  b2_2=b2_2[(1:length(X[,1])),]
  b2_3=b2_3[(1:length(X[,1])),]
  b2_4=b2_4[(1:length(X[,1])),]
  
  
  W3_1 <-para[[29]]
  W3_2 <-para[[30]]  
  W3_3 <-para[[31]]
  W3_4 <-para[[32]]
  
  b3_1 <-para[[33]]
  b3_1=b3_1[(1:length(X[,1]))]
  
  
  
  
  
  xt=X
  N <- nrow(X)
  M <- ncol(xt) # number of timepoints for xt
  
  
  
  
  vi1 <- seq(0, 1, by=1/(M-1))
  vi_v <- t(as.matrix(bSpline(vi1,df=vi)))
  
  
  vj1 <- seq(0, 1, by=1/(S-1))
  vj_v <- t(as.matrix(bSpline(vj1,df=vj)))
  
  vk1 <- seq(0, 1, by=1/(S-1))
  vk_v <- t(as.matrix(bSpline(vk1,df=vk)))
  
  vm1 <- seq(0, 1, by=1/(S-1))
  vm_v <- t(as.matrix(bSpline(vm1,df=vm)))
  
  vn1 <- seq(0, 1, by=1/(S-1))
  vn_v <- t(as.matrix(bSpline(vn1,df=vn)))
  
  
  
  ai1_1=t(as.matrix(vi_v)%*%t(xt)/M)
  
  
  awv1_1.1=ai1_1%*%W1_1
  awv1_1=awv1_1.1%*%vj_v
  
  hidden_layer11 <- relu(b1_1+awv1_1)
  hidden_layer11<- matrix(hidden_layer11, nrow  = N)
  
  
  
  awv1_2.1=ai1_1%*%W1_2
  awv1_2=awv1_2.1%*%vj_v
  
  hidden_layer12 <- relu(b1_2+awv1_2)
  hidden_layer12<- matrix(hidden_layer12, nrow  = N)
  
  awv1_3.1=ai1_1%*%W1_3
  awv1_3=awv1_3.1%*%vj_v
  
  hidden_layer13 <- relu(b1_3+awv1_3)
  hidden_layer13<- matrix(hidden_layer13, nrow  = N)
  
  awv1_4.1=ai1_1%*%W1_4
  awv1_4=awv1_4.1%*%vj_v
  
  hidden_layer14 <- relu(b1_4+awv1_4)
  hidden_layer14<- matrix(hidden_layer14, nrow  = N)
  
  
  #layer 2
  ai2_11=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_11.1=ai2_11%*%W2_11
  awv2_11=awv2_11.1%*%vn_v
  
  ai2_21=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_21.1=ai2_21%*%W2_21
  awv2_21=awv2_21.1%*%vn_v
  
  ai2_31=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_31.1=ai2_31%*%W2_31
  awv2_31=awv2_31.1%*%vn_v
  
  ai2_41=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_41.1=ai2_41%*%W2_41
  awv2_41=awv2_41.1%*%vn_v
  
  
  hidden_layer21 <- relu(b2_1+awv2_11+awv2_21+awv2_31+awv2_41)
  hidden_layer21<- matrix(hidden_layer21, nrow  = N)
  
  
  
  ai2_12=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_12.1=ai2_12%*%W2_12
  awv2_12=awv2_12.1%*%vn_v
  
  ai2_22=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_22.1=ai2_22%*%W2_22
  awv2_22=awv2_22.1%*%vn_v
  
  ai2_32=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_32.1=ai2_32%*%W2_32
  awv2_32=awv2_32.1%*%vn_v
  
  ai2_42=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_42.1=ai2_42%*%W2_42
  awv2_42=awv2_42.1%*%vn_v
  
  
  hidden_layer22 <- relu(b2_2+awv2_12+awv2_22+awv2_32+awv2_42)
  hidden_layer22<- matrix(hidden_layer22, nrow  = N)
  
  
  
  ai2_13=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_13.1=ai2_13%*%W2_13
  awv2_13=awv2_13.1%*%vn_v
  
  ai2_23=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_23.1=ai2_23%*%W2_23
  awv2_23=awv2_23.1%*%vn_v
  
  ai2_33=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_33.1=ai2_33%*%W2_33
  awv2_33=awv2_33.1%*%vn_v
  
  ai2_43=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_43.1=ai2_43%*%W2_43
  awv2_43=awv2_43.1%*%vn_v
  
  
  hidden_layer23 <- relu(b2_3+awv2_13+awv2_23+awv2_33+awv2_43)
  hidden_layer23<- matrix(hidden_layer23, nrow  = N)
  
  
  ai2_14=t(as.matrix(vm_v)%*%t(hidden_layer11)/S)
  
  awv2_14.1=ai2_14%*%W2_14
  awv2_14=awv2_14.1%*%vn_v
  
  ai2_24=t(as.matrix(vm_v)%*%t(hidden_layer12)/S)
  
  awv2_24.1=ai2_24%*%W2_24
  awv2_24=awv2_24.1%*%vn_v
  
  ai2_34=t(as.matrix(vm_v)%*%t(hidden_layer13)/S)
  
  awv2_34.1=ai2_34%*%W2_34
  awv2_34=awv2_34.1%*%vn_v
  
  ai2_44=t(as.matrix(vm_v)%*%t(hidden_layer14)/S)
  
  awv2_44.1=ai2_44%*%W2_44
  awv2_44=awv2_44.1%*%vn_v
  
  
  hidden_layer24 <- relu(b2_4+awv2_14+awv2_24+awv2_34+awv2_44)
  hidden_layer24<- matrix(hidden_layer24, nrow  = N)
  
  #last layer
  ai3_1=t(as.matrix(vk_v)%*%t(hidden_layer21)/S)
  
  awv3_1=ai3_1%*%W3_1
  
  
  ai3_2=t(as.matrix(vk_v)%*%t(hidden_layer22)/S)
  
  awv3_2=ai3_2%*%W3_2
  
  ai3_3=t(as.matrix(vk_v)%*%t(hidden_layer23)/S)
  
  awv3_3=ai3_3%*%W3_3
  
  
  ai3_4=t(as.matrix(vk_v)%*%t(hidden_layer24)/S)
  
  awv3_4=ai3_4%*%W3_4
  
  
  
  hidden_layer3 <- (b3_1+awv3_1+awv3_2+awv3_3+awv3_4)
  hidden_layer3<- matrix(hidden_layer3, nrow  = N)
  
  
  
  # compute the loss: sofmax and regularization
  yhat=hidden_layer3
  
  return(yhat)  
}


fbnn.model <- fbnn4.4.es(X=xt, Y=Y,XV=xtv,YV=yv, S=30, vi=5, vj=5, vk=5,vm=5,vn=5, step_size=.001, niteration = 5000)



t4_i21=rmse(as.numeric(y),as.numeric(fbnn.pred4.4(X=xt, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)))
t4_i21

fbnn.pred.model <- fbnn.pred4.4(X=xtv, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)
t4p_i21v=rmse(as.numeric(yv),as.numeric(fbnn.pred.model))

fbnn.pred.model <- fbnn.pred4.4(X=xtp, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)
t4p_i21=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
t4_i21
t4p_i21v
t4p_i21

