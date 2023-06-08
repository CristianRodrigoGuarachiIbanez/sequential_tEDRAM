from plotille import Figure
from numpy import ndarray
import numpy as np

class TerminalGraphics(object):
    fig:Figure
    def __init__(self, w:int, h:int, limits_x:tuple, limits_y:tuple)->None:
        self.setup_graphic_parameters(w, h, limits_x, limits_y)
    def setup_graphic_parameters(self,w:int, h:int, limits_x:tuple, limits_y:tuple)->None:
        self.fig = Figure()
        self.fig.width = w
        self.fig.height = h
        self.fig.set_x_limits(min_=limits_x[0], max_=limits_x[1])
        self.fig.set_y_limits(min_=limits_y[0], max_=limits_y[1])
        self.fig.color_mode = 'byte'
    def plot_config(self, X:ndarray, Y:ndarray, c:int, l:str )->None:
        self.fig.plot(X, Y, lc=c, interp='linear', label=l)
    def histogram(self, X:ndarray, color:int)->None:
        self.fig.histogram(X, bins=160, lc=color)
    def show(self)->None:
        print(self.fig.show(legend=True))
# fig = Figure()
# fig.width = 100
# fig.height = 2
# fig.set_x_limits(min_=0, max_=10)
# fig.set_y_limits(min_=0, max_=2)
# fig.color_mode = 'byte'
# #fig.plot([-0.5, 1], [-1, 1], lc=25, label='First line')
# fig.histogram(X, bins= 160,lc=100)
# #fig.plot(X, (X+2) , lc=200, label='square')
# print(fig.show(legend=True))

if __name__ == '__main__':
    X = np.sort(np.arange(50))
    Y = np.sort(np.linspace(0,10))
    print(X.shape,Y.shape)
    gr = TerminalGraphics(100, 100, (0,10), (0,10))
    #gr.histogram(X)
    gr.plot_config(X,Y)
    gr.show()