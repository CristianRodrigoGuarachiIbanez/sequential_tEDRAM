from numpy import asarray, int32, zeros, all
from os import getcwd
from pickle import dump, load

class TXTHandler:
    def __init__(self, filename:str):
        self._open_labels(filename)
    def _open_labels( self, filename: str):
        with open(filename, "rb") as file:
            self.grTruth = asarray(load(file), dtype=int32)
            file.close()
    def save(self, data, out="./labels.txt"):
        dump(data, open(out, "wb"))
    def convert6to4categories(self, out=4):
        len_1 = self.grTruth.shape[0]
        len_2 =self.grTruth.shape[1]
        output = zeros((len_1, out), dtype=int32)
        for i in range(len_1):
            if (all(self.grTruth[i] == 0)):
                #print(" ejemplo -> ", self.grTruth[i])
                output[i, 3] = 1
            for j in range(len_2):
                if(self.grTruth[i,j]==1 and j==0):
                    output[i,0] = self.grTruth[i,j]
                elif(self.grTruth[i,j]==1 and j==1):
                    output[i,0] =self.grTruth[i,j]
                elif (self.grTruth[i, j] == 1 and j == 2):
                    #print(self.grTruth[i, j])
                    output[i, 1] = self.grTruth[i, j]
                elif (self.grTruth[i, j] == 1 and j == 3):
                    output[i, 1] = self.grTruth[i, j]
                elif (self.grTruth[i, j] == 1 and j == 4):
                    output[i, 2] = self.grTruth[i, j]
                elif (self.grTruth[i, j] == 1 and j == 5):
                    output[i, 2] = self.grTruth[i, j]

        #print("test ->", output.shape, output[2030:2130,:])
        self.save(output)

    def numberOfImagesTillCollision(self, out=1):
        len_1 = self.grTruth.shape[0]
        output = zeros((len_1, out), dtype=int32)
        counter = 0
        for i in range(len_1):
            if(all(self.grTruth[i]==0)):
                counter +=1
                output[i]=counter
            else:
                counter = 0
                output[i]=counter
        print("test -> ", output[40000:40100])
        self.save(output, out = "./labels_counting.txt")
if __name__ == '__main__':
    path = r"/label_data.txt"
    dv = TXTHandler(filename=getcwd() + path)
    #dv.convert6to4categories()
    dv.numberOfImagesTillCollision()
