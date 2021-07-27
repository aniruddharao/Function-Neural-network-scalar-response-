library(fds)
library(ggplot2)
library(reshape2)
library(MASS)
library(mice)
library(lattice)
library(missForest)
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

#this file contains 3 network frameworks solved using fdnn and fbnn.

#value of hyperparameter S, vi, vk, vk, vn and vm don't affect the performance much.

# main parameter that needs to be tuned is the step size and number of iteration so that the model does not get stuck
#at a local minima or overfit.

#sometimes the algorigthm gets stuck at a local minima because of the seed (initialization of the random funcitons)



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
y=yy[1:180]
xt=x1[1:180,]




#test
yp=yy[-(1:180)]
xtp=x1[-(1:180),]



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


#the function below fits our network using FDNN strategy. It gives out the parameter values, loss and predict value.
fdnn <- function(X, Y, S, step_size, niteration){
  
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
  
  # gradient descent loop to update weight and bias
  for (i in 0:niteration){
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
    loss1=rmse(as.numeric(Y),as.numeric(yhat))
    
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
  }
  return(list(W, W2, b, b2,sqrt(loss),yhat))
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


fdnn.model <- fdnn(X=xt, Y=Y, S=30, step_size = 0.01, niteration = 10000)

#training loss
fdnn.pred.model <- fdnn.pred(X=xt, S=30, fdnn.model)
t44=rmse(as.numeric(y),as.numeric(fdnn.pred.model))


#testing error
fdnn.pred.model <- fdnn.pred(X=xtp, S=30, fdnn.model)
t4p4=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
t44
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

#vi, vj, vk are the number of basis functions (bspline) used in the hidden layers.

#the function below fits our network using FBNN strategy. It gives out the parameter values, loss and predict value.


Y=y
X=xt





#the function below fits our network using FBNN strategy. It gives out the parameter values, loss and predict value.
fbnn <- function(X, Y, S, vi, vj, vk, step_size = 0.5, niteration){
  
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
  
  
  
  
  # initialize parameters randomly
  W <- 0.01 * matrix(rnorm(vi*vj), nrow = vi)
  b <- matrix(0, nrow = N, ncol = S)
  W2 <- 0.01 * as.vector(rnorm(vk))
  b2 <- matrix(0, nrow = N, ncol = 1)
  
  
  # gradient descent loop to update weight and bias
  for (r in 0:niteration){
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
    
    
    # check progress
    if (r%%1000 == 0 | r == niteration){
      print(paste("iteration", r,': loss', loss))}
    
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
  }
  return(list(W, W2, b, b2, sqrt(loss)))
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


fbnn.model <- fbnn(X=xt, Y=Y, S=30, vi=5, vj=5, vk=5, step_size=.01, niteration = 10000)


#training error
t4_i2=rmse(as.numeric(y),as.numeric(fbnn.pred(X=xt, S=30, vi=5, vj=5, vk=5, fbnn.model)))
t4_i2



#testing error
fbnn.pred.model <- fbnn.pred(X=xtp, S=30, vi=5, vj=5, vk=5, fbnn.model)
t4p_i2=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
t4_i2
t4p_i2












####################################
#########FdNN22
####################################


#Here we are defining 2 hidden layers with 2 neurons each and using fdnn strategy


fdnn2.2 <- function(X, Y, S, step_size, niteration){
  
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
    
    
    
    
  }
  return(list(W1_1, W1_2, b1_1, b1_2, 
              W2_11, W2_12, W2_21, W2_22, b2_1, b2_2, 
              W3_1, W3_2, b3_1,
              sqrt(loss),yhat))
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




fdnn.model <- fdnn2.2(X=xt, Y=Y, S=30, step_size = 0.01, niteration = 10000)

t44=rmse(as.numeric(y),as.numeric(fdnn.pred2.2(X=xt, S=30, fdnn.model)))
t44


fdnn.pred.model <- fdnn.pred2.2(X=xtp, S=30, fdnn.model)
t4p4=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
t44
t4p4





####################################
#########FdNN44
####################################


#Here we are defining 2 hidden layers with 4 neurons each and using fdnn strategy


fdnn4.4 <- function(X, Y, S, step_size, niteration){
  
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
    
    
    
    
    
  }
  return(list(W1_1, W1_2, W1_3, W1_4, b1_1, b1_2, b1_3, b1_4, 
              W2_11, W2_12, W2_13, W2_14, W2_21, W2_22, W2_23, W2_24, W2_31, W2_32, W2_33, W2_34, W2_41, W2_42, W2_43, W2_44,
              b2_1, b2_2, b2_3, b2_4, 
              W3_1, W3_2, W3_3, W3_4, b3_1,
              sqrt(loss),yhat))
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



fdnn.model <- fdnn4.4(X=xt, Y=Y, S=30, step_size = 0.01, niteration = 10000)

t441=rmse(as.numeric(y),as.numeric(fdnn.pred4.4(X=xt, S=30, fdnn.model)))
t441


fdnn.pred.model <- fdnn.pred4.4(X=xtp, S=30, fdnn.model)
t4p41=rmse(as.numeric(yp),as.numeric(fdnn.pred.model))
t441
t4p41






####################################
#########FBNN22
####################################


#Here we are defining 2 hidden layers with 2 neurons each and using fBnn strategy

#vm and vn are the number of basis functions (bspline) used in the additional hidden layers.




fbnn2.2 <- function(X, Y, S, vi, vj, vk,vm,vn, step_size = 0.5, niteration){
  
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
    
    
    
    
    
  }
  return(list(W1_1,W1_2,b1_1,b1_2,
              W2_11,W2_12,W2_21,W2_22, b2_1, b2_2,
              W3_1,W3_2,b3_1,sqrt(loss)))
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



fbnn.model <- fbnn2.2(X=xt, Y=Y, S=30, vi=5, vj=5, vk=5,vm=5,vn=5, step_size=.001, niteration = 10000)



t4_i2=rmse(as.numeric(y),as.numeric(fbnn.pred2.2(X=xt, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)))
t4_i2



fbnn.pred.model <- fbnn.pred2.2(X=xtp, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)
t4p_i2=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
t4_i2
t4p_i2





####################################
#########FBNN44
####################################


#Here we are defining 2 hidden layers with 4 neurons each and using fBnn strategy

#vm and vn are the number of basis functions (bspline) used in the additional hidden layers.
fbnn4.4 <- function(X, Y, S, vi, vj, vk,vm,vn, step_size = 0.5, niteration){
  
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
    
    
    
    
    
  }
  return(list(W1_1,W1_2,W1_3,W1_4,b1_1,b1_2,b1_3,b1_4,
              W2_11,W2_12,W2_13,W2_14,W2_21,W2_22,W2_23,W2_24,W2_31,W2_32,W2_33,W2_34,W2_41,W2_42,W2_43,W2_44,b2_1,b2_2,b2_3,b2_4,
              W3_1,W3_2,W3_3,W3_4,b3_1,sqrt(loss)))
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



fbnn.model <- fbnn4.4(X=xt, Y=Y, S=30, vi=5, vj=5, vk=5,vm=5,vn=5, step_size=.001, niteration = 10000)



t4_i21=rmse(as.numeric(y),as.numeric(fbnn.pred4.4(X=xt, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)))
t4_i21



fbnn.pred.model <- fbnn.pred4.4(X=xtp, S=30, vi=5, vj=5, vk=5,vm=5,vn=5,  fbnn.model)
t4p_i21=rmse(as.numeric(yp),as.numeric(fbnn.pred.model))
t4_i21
t4p_i21

