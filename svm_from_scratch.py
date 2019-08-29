#!/usr/bin/env python
# coding: utf-8

# In[111]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


# In[145]:


style.use('ggplot')

class svm:
    def __init__(self,visualize=False):
        self.visualize=visualize
        self.colors = {-1:'r',1:'b'}
        if self.visualize:
            self.fig = plt.figure()
            self.axis = self.fig.add_subplot(1,1,1)
    def train(self,data):
            self.data = data
            #keeping w and b amount in every opoch: opt_dict = {|w| :[w,b]}
            opt_dic = {}
            #transforms for creating different w with the same |w|(because we gotta check all of them !)
            transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
            #saving w and b values for stepping
            all_data = []
            for yi in self.data:
                for featureset in self.data[yi]:
                    for feature in featureset:
                        all_data.append(feature)
            self.max = max(all_data)
            self.min = min(all_data)
            #freeing the all_data from memory
            all_data = None
            step_sizes = [self.max*0.1,self.max*0.01,self.max*0.001]
            #b step sizes
            b_range_step = 5
            b_step = 5
            latest_optimum = self.max * 10 #starting definition of vector w
            for step in step_sizes:
                w = np.array([latest_optimum,latest_optimum-20])
                optimized = False
                while not optimized:
                    for b in np.arange(-1*self.max*b_range_step,self.max*b_range_step,step*b_step):#third value: step size for arange
                        for transformation in transforms:
                            w_transformed = w*transformation
                            found_option = True
                            for yi in self.data:
                                for xi in self.data[yi]:
                                    if not yi*(np.dot(xi,w_transformed)+b)>=1:
                                        found_option = False
                                        break # not nesessary to check others since we already got the wrong w and b
                            if found_option:
                                opt_dic[np.linalg.norm(w_transformed)] = [w_transformed,b]
                    if(w[0]<0):
                        optimized = True
                        print('optimized a step')
                    else:
                        w = w-step
                norms = sorted([n for n in opt_dic])
                #opt_dict : |w|: [w,b]
                opt_choice = opt_dic[norms[0]]
                self.w = opt_choice[0]
                self.b = opt_choice[1]
                print(opt_choice)
                latest_optimum = opt_choice[0][0] + step *2
            for i in self.data:
                for xi in self.data[i]:
                    yi=i
                    print(xi,':',yi*(np.dot(self.w,xi)+self.b))
    def predict(self,features):
            clf = np.sign(np.dot(np.array(features),np.array(self.w))+self.b)
            if clf !=0 and self.visualize:
                self.axis.scatter(features[0],features[1],s=200,marker='*',c=self.colors[clf])
            return clf
    def plot_svm(self):
        #scattering known featuresets.
        [[self.axis.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data[i]] for i in data]
        def hyperplane(x,w,b,v):
            # v = (w.x+b)
            return (-w[0]*x-b+v) / w[1]
        datarange = (self.min*0.9,self.max*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # w.x + b = 1
        # pos sv hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.axis.plot([hyp_x_min,hyp_x_max], [psv1,psv2], "k")
        # w.x + b = -1
        # negative sv hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.axis.plot([hyp_x_min,hyp_x_max], [nsv1,nsv2], "k")

        # w.x + b = 0
        # decision
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.axis.plot([hyp_x_min,hyp_x_max], [db1,db2], "g--")
        plt.show()


# In[146]:


mysvm = svm(visualize=True)
data = {-1:[[1,3],[2,4],[3,5]],1:[[5,7],[6,8],[7,9]]}
xs = []
ys = []
color = []


# In[147]:


mysvm.train(data)


# In[138]:


#mysvm.plot_svm()


# In[140]:


pred = [[0,10],[1,3],[3,4]]
res = []
for p in pred:
    res.append(mysvm.predict(p))
print(res)


# In[141]:


mysvm.plot_svm()


# In[ ]:




