import copy
import math
import time

from Pyro4 import expose
class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers
        self.result = ""
        self.tb=2.326
        self.p1=0.25
        self.p2=0.5
        self.ta=2.336
        self.n1=9
        self.n2=9
        self.n3=9
        self.length=2048


    def solve(self):
        inpt = self.read_input()
        temp = pow((self.tb*self.p2*(1-self.p2)+self.ta*math.sqrt(self.p1*(1-self.p1)))/(self.p2-self.p1),2)
        N = int(temp)
        C = int(temp*self.p1+self.ta*math.sqrt(temp*self.p1*(1-self.p1)))
        self.result+=str(N)+"\n"+str(C)+"\n"
        x=[]
        for i in range(0,self.n1):
            x.append(0)
        y = []
        for i in range(0, self.n2):
            y.append(0)
        mapped1 = []
        n1 = 2**self.n1 / (len(self.workers))

        for i in range(0,(len(self.workers))):
            tempx = copy.deepcopy(x)
            mapped1.append(self.workers[i].predict_x(self.next_var(tempx,self.n1,n1*i),n1*i,n1*i+n1,inpt,C,N))



        arr_x= self.collect1(mapped1)
        if arr_x is not None:
            self.result+=str(len(arr_x))+"\n"
            self.result+=str([1,0,1,0,1,0,1,1,1] in arr_x)+"\n"
        mapped2 = []
        n2 = 2 ** self.n2 / (len(self.workers))
        for i in range(0, (len(self.workers))):
            mapped2.append(
                self.workers[i].predict_y(self.next_var(copy.deepcopy(y), self.n2, n2 * i), n2 * i, n2 * i + n2, inpt,
                                          C, N))
        arr_y = self.collect1(mapped2)
        if arr_y is not None:
            self.result += str(len(arr_y)) + "\n"
            self.result+=str([0,0,0,1,1,0,1,0,1] in arr_y) + "\n"

        step = len(arr_y)/(len(self.workers))
        mapped3 = []
        for i in range(0, (len(self.workers))):
            mapped3.append(self.workers[i].predict_s(arr_y[step * i:i * step + step], arr_x, inpt))


        result = self.collect2(mapped3)

        if result != None:
            for el in result:
                self.result+=str(el)+"\n"

        self.write_output(self.result)
        return 0


    @expose
    def predict_x(self,x, n1,n2, line, C, N):
        res =[]
        for i in range(n1,n2):
            R = self.calc_R_x(line[0:N], x, N)
            if R < C:
                res.append(copy.deepcopy(x))
            self.next_var(x,self.n1,1)
        return res

    @expose
    def predict_y(self, y, n1, n2, line, C, N):
        res = []
        for i in range(n1, n2):
            R = self.calc_R_y(line[0:N], y, N)
            if R < C:
                res.append(copy.deepcopy(y))
            self.next_var(y, self.n1, 1)
        return res

    @expose
    def predict_s(self, y, x,line):
        for ely in y:
            for elx in x:

                s_candidates = self.generate_s(elx,ely,line)
                for els in s_candidates:
                    check = self.Geffe(elx,ely,els)

                    if check==line:
                        z =[]
                        for i in range(0,self.n3):
                            z.append(int(els[i]))
                        return [elx,ely,z]

    @staticmethod
    @expose
    def collect2(mapped):
        res = []
        for el in mapped:
            temp = el.value
            if temp is None:
                continue
            return temp
        return None

    @staticmethod
    @expose
    def collect1(mapped):
        res = []
        for el in mapped:
            res.extend(el.value)
        return res


    @staticmethod
    @expose
    def mymap(a, b):
        print (a, b)
        res = 0
        for i in range(a, b):
            res += i
        return res


    def read_input(self):
        f = open(self.input_file_name, 'r')
        line = f.readline()
        f.close()
        return line
    def write_output(self, output):
        f = open(self.output_file_name, 'w')
        f.write(str(output))
        f.write('\n')
        f.close()

    def L1(self, state):
        next = (state[0]+state[3])%2
        zero = state[0]
        for i in range(0,self.n1-1):
            state[i]=state[i+1]
        state[self.n1-1]=next
        return zero

    def L2(self, state):
        next = (state[0] + state[1] + state[2] + state[6])%2
        zero = state[0]
        for i in range(0,self.n2-1):
            state[i]=state[i+1]
        state[self.n2-1]=next
        return zero

    def L3(self, state):
        next = (state[0] + state[1] + state[2] + state[5])%2
        zero = state[0]
        for i in range(0,self.n3-1):
            state[i]=state[i+1]
        state[self.n3-1]=next
        return zero

    def Geffe(self, lx1,ly1,lz1):
        x = []
        for i in range(0,len(lx1)):
            x.append(int(lx1[i]))
        y = []
        for i in range(0, len(ly1)):
            y.append(int(ly1[i]))
        z = []
        for i in range(0, len(lz1)):
            z.append(int(lz1[i]))
        res =""
        for i in range(0,self.length):
            res+=str((z[0]*x[0]+(1+z[0])%2*y[0])%2)
            self.L1(x)
            self.L2(y)
            self.L3(z)
        return res

    def calc_R_x(self,s,x1,N):
        R = 0
        x = copy.deepcopy(x1)
        for i in range(0,N):
            R+=(x[0]+int(s[i]))%2
            self.L1(x)
        return R

    def calc_R_y(self,s,y1,N):
        R =0
        y = copy.deepcopy(y1)
        for i in range(0,N):
            R+=(y[0]+int(s[i]))%2
            self.L2(y)
        return R


    def next_var(self, arr, l,n):
        for i in range(0,n):
            temp = 1
            for j in range(0,l):
                arr[j]+=temp
                temp = int(arr[j]/2)
                arr[j]%=2
                if temp==0:
                    break
        return arr

    def generate_x(self, x):
        res =""
        for i in range(0,self.length):
            res+=str(x[0])
            self.L1(x)
        return res

    def generate_y(self, y):
        res =""
        for i in range(0,self.length):
            res+=str(y[0])
            self.L1(y)
        return res

    def generate_s(self,x1,y1,line):
        res =[]

        res.append("")
        i = 0
        x = copy.deepcopy(x1)
        y = copy.deepcopy(y1)
        for j in range(0,self.n3):
            if x[i]==y[i] and x[i]==int(line[j]):
                restemp=[]
                for el in res:
                    restemp.append(el + "0")
                    restemp.append(el + "1")
                res = restemp
            elif x[i]==y[i] and x[i]!=int(line[j]):
                res = []
                return res
            elif x[i]==1 and line[j]=="1":
                restemp = []
                for el in res:
                    restemp.append(el + "1")
                res = restemp
            elif x[i]==0 and line[j]=="0":
                restemp = []
                for el in res:
                    restemp.append(el + "1")
                res = restemp
            elif y[i]==1 and line[j]=="1":
                restemp = []
                for el in res:
                    restemp.append(el + "0")
                res = restemp
            elif y[i]==0 and line[j]=="0":
                restemp = []
                for el in res:
                    restemp.append(el + "0")
                res = restemp
            else:
                res=[]
                return res
            self.L1(x)
            self.L2(y)
        return res

"""if __name__=="__main__":
    test = Solver()
    test.solve()
"""