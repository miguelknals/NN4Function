import math 
import os
import random

PI=3.141592

class MyFunctionData():
    
    def __init__(self,  file_name,nvalues):
        super().__init__()
        #
        # x range 
        #        
        x_l=[]; f_l=[]
        for i in range (1,nvalues):
            # x range between 0 and 1 * PI
            # uniform same probability distribution
            #
            x= [PI*random.uniform(0,1), PI*random.uniform(0,1), 
            PI*random.uniform(0,1), PI*random.uniform(0,1)]
            f= self.multif(x) # we evaluate the function
            x_l.append(x)     # we store x    
            f_l.append(f)     # we store f(x)
        
        # we store values
        of1=os.path.join(file_name)
        with open(of1, encoding='utf-8', mode ='w') as ofile1:
            for x,f in zip(x_l,f_l):
                ofile1.write("{}\t{}\t{}\t{}\t{}\n".format(
                    x[0],x[1],x[2],x[3],
                    f))

    def multif(self, x):
        f1= math.sin (x[0])
        f2= math.sin (x[1]*2)/2
        f3= math.sin (x[2]*4)/4
        f= (f1+f2+f3) * x[3]
        #f=4.0*math.sin(x[0]*x[0] +2.0* x[1]*x[1]) * (1.0+x[2]) +(1.0+x[3]*x[3]*x[0])
        #f=(x[0]*x[0] +2.0* x[1]*x[1]) * (1.0+x[2]) +(1.0+x[3]*x[3]) # f=(x0^2 + 2*x1^2) *(1+x2) + (1+x3^2)
        return f

