''' Feel free to use numpy for matrix multiplication and
	other neat features.
	You can write some helper functions to
	place some calculations outside the other functions
	if you like to.
'''
import numpy as np

class mlp:
	def __init__(self, inputs, targets, nhidden):
                """
                Initialize MLP
                """
		self.beta = 1
		self.eta = 0.1
		self.momentum = 0.0

                #No. of input, hidden and output layer nodes respectively
                self.ninput = inputs.shape[1]
                self.nhidden = nhidden
                self.noutput = targets.shape[1]

                self.ndata = np.shape(inputs)[0]

                #Weights from Input to Hidden layer
                self.V = (np.random.rand(self.ninput+1,self.nhidden)-0.5)*2/np.sqrt(self.ninput)

                #Weights from Hidden to Output layer
                self.W = (np.random.rand(self.nhidden+1,self.noutput)-0.5)*2/np.sqrt(self.nhidden)

	def earlystopping(self, inputs, targets, valid, validtargets, iterations=100 ):
                """
                Starts the training of the network and keeps track when to stop the training.
                """

                valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)

                old_val_error1 = 100002
                old_val_error2 = 100001
                new_val_error = 100000

                while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
                        self.train(inputs,targets,iterations)
                        old_val_error2 = old_val_error1
                        old_val_error1 = new_val_error
                        validout = self.forward(valid)
                        new_val_error = 0.5*np.sum((validtargets-validout)**2)
                        
#                print "Stopped", new_val_error,old_val_error1, old_val_error2
                return new_val_error


	def train(self, inputs, targets, iterations=100):
                """
                Trains the MLP network.
                """
                inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
                change = range(np.shape(inputs)[1])
                
                updateV = np.zeros((np.shape(self.V)))
                updateW = np.zeros((np.shape(self.W)))

                count = 0
                for index in range(iterations):
                        count += 1
                        self.output = self.forward(inputs)

                        error = 0.5*np.sum((self.output-targets)**2)

#                        if (np.mod(index,100)==0):
#                                print "Iteration: ", index , " Error: ",error


                        #for softmax activation 
                        deltao = (self.output-targets)*(self.output*(-self.output)+self.output)/self.ndata


                        deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.W)))

                        updateV = self.eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updateV
                        updateW = self.eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updateW
                        self.V -= updateV
                        self.W -= updateW


	def forward(self, inputs):
                """
                Run the MLP network forward
                """
                #Get the hidden layer neurons
                self.hidden = np.dot(inputs,self.V)
                self.hidden = 1.0/(1.0 + np.exp(-self.beta * self.hidden))
                self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

                #Get the output layer neurons
                output = np.dot(self.hidden,self.W)

                #soft-max
                normalisers = np.sum(np.exp(output),axis=1)*np.ones((1,np.shape(output)[0]))
                return np.transpose(np.transpose(np.exp(output))/normalisers)



	def confusion(self, inputs, targets):
                """
                Prints a confusion matrix.
                """

                cm = np.zeros((self.noutput,self.noutput))
                
                inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
                output = self.forward(inputs)

                #no. of classes
                nclasses = np.shape(targets)[1]
                
                output = np.argmax(output,1)
                targets = np.argmax(targets,1)

                for i in range(nclasses):
                        for j in range(nclasses):
                                cm[i,j] = np.sum(np.where(output==i,1,0)*np.where(targets==j,1,0))

                total = np.sum(cm)
                correct = np.trace(cm)
                percent = correct/total * 100

                print "Confusion matrix : "
                print cm
                print "Percentage Correct : ", percent
                return cm
                









