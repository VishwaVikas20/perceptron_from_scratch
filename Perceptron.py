import numpy as np
def step_function(x):
    return np.where(x>0,1,0)
class Perceptron:
    def __init__(self,learning_rate=0.01,n_iters=1000):
        self.learning_rate=learning_rate
        self.n_iters=n_iters
        self.activation_function=step_function
        self.weights=None
        self.bias=None
    def train(self,X,y):
        n_samples,n_features=X.shape
        #intialize weights and bias
        self.weights=np.zeros(n_features)
        self.bias=0
        y_=np.where(y>0,1,0)
        for _ in range(self.n_iters):
            for idx,X_i in enumerate(X):
                weighted_sum=np.dot(X_i, self.weights)+self.bias
                pred=self.activation_function(weighted_sum)
                error=y[idx]-pred
                #This is the actual training.
                self.weights+=error*X_i*self.learning_rate
                self.bias+=error*self.learning_rate
        
    def predict(self,X):
        weighted_sum=np.dot(self.weights,X)+self.bias
        prediction=self.activation_function(weighted_sum)
        return prediction
def print_table(func,name:str):
  print("Printing Truth Table of "+name)
  for X_i in X:
      print("-"*20)
      print(f"| {X_i[0]}  |   {X_i[1]}  |   {func.predict(X_i)}  |")
  print("-"*20)
  print("\n\n\n")
X=[
    [0,1],
    [1,0],
    [0,0],
    [1,1]
    ]
y_AND=[0,0,0,1]
y_OR=[1,1,0,1]
y_NOR=[0,0,1,0]

AND=Perceptron()
OR=Perceptron()
NOR=Perceptron()
AND.train(np.array(X),np.array(y_AND))
OR.train(np.array(X),np.array(y_OR))
NOR.train(np.array(X),np.array(y_NOR))

print_table(AND,"AND_Gate")
print_table(OR,"OR_Gate")
print_table(NOR,"NOR_Gate")

    
