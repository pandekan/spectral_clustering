import numpy as np
import graph_utilities
import h5py
import time
import sys
import matplotlib.pyplot as plt
import clustering
import pyqtgraph as pg
try:
    from PyQt5 import QtCore, QtGui
    from PyQt5.uic import loadUiType
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4 import QtCore, QtGui
    from PyQt4.uic import loadUiType
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

#from PyQt5.QtGui import *
#from PyQt5.QtWidgets import *
#from PyQt5.QtCore import pyqtSignal as Signal

#fname = sys.argv[1]
#feature = sys.argv[2]
#ndata = int(sys.argv[3])
#sigma = float(sys.argv[4])
#nclusters = int(sys.argv[5])

lastclicked=[]

class ZoomScatterItem(pg.ScatterPlotItem):
    sigShowCart = Signal(int)

    def __init__(self, *args, **kwargs):
        super(ZoomScatterItem, self).__init__(*args, **kwargs)
        self.sigClicked.connect(self.clicked)

    def clicked(self,_ , points):
        global lastclicked
        for p in lastclicked:
            p.resetPen()
        for point in points:
            point.setPen('b',width=2)
            index = point.data()
            #print('index:',index)
            #print point.pos()
            self.sigShowCart.emit(index)
        lastclicked=points

    def setImageIndices(self, indices):
        self.imageindices=indices
        # self.mousePoint = self.viewBox.mapSceneToView(position)
        # print(self.mousePoint.toTuple())

class ExportROI(pg.RectROI):
    def __init__(self, bounditem, *args, **kwargs):
        super(ExportROI, self).__init__(*args, **kwargs)
        self.sigRegionChangeFinished.connect(self.export)
        self.bounditem=bounditem # type: pg.ScatterPlotItem

    def export(self):
        exportpoints = []
        for point in self.bounditem.points():
            if point.pos().x()>self.pos().x() \
                and point.pos().y()>self.pos().y() \
                and point.pos().x()<self.size().x()+self.pos().x() \
                and point.pos().y()<self.size().y()+self.pos().y():
                exportpoints.append(point.data())
        print exportpoints

        return exportpoints

def main(fname,*args,**kwargs):

    return fname, kwargs #fname, eigvec, nclusters

if __name__ == "__main__":

    #eigvec, select_ind, nclusters, fname  = main(*args)

    #fname, eigvec, nclusters = main(sys.argv[1], **kwargs)
    #print fname, nclusters

    #fname, kwargs = main(sys.argv[1], kwargs)
    #print fname

    
    app = QApplication([])
    mainwindow = QMainWindow()
    splitter = QSplitter()
    mainwindow.setCentralWidget(splitter)
    grid = QGridLayout()
    leftwidget = QWidget()
    leftwidget.setLayout(grid)
    splitter.addWidget(leftwidget)

    eigen12 = pg.PlotWidget()
    eigen13 = pg.PlotWidget()
    eigen23 = pg.PlotWidget()
    eigen14 = pg.PlotWidget()
    cart_view = pg.ImageView()
    grid.addWidget(eigen12, 0, 0, 1, 1)
    grid.addWidget(eigen13, 0, 1, 1, 1)
    grid.addWidget(eigen23, 1, 0, 1, 1)
    grid.addWidget(eigen14, 1, 1, 1, 1)
    splitter.addWidget(cart_view)

    def showframe(index):
        cart_view.setImage(f['data/cart_data'][index])

    def saveROI(indices):
        
        fout = h5py.File('save_0.h5','w')
        npatt = np.size(indices)
        ny, nx = f['data/cart_data'][indices[0]].shape
        dset = fout.create_dataset('data/data',(npatt,ny,nx),dtype=float)

        for i in range(npatt):
            dset[i,:,:] = f['data/cart_data'][indices[i],:,:]

        fout.create_dataset('data/timestamps',data=f['data/timestamps'][indices])

        fout.close()

    def avgROI(indices):

        avg = np.average(f['data/cart_data'][indices,:,:],axis=0)
        cart_view.setImage(avg)


    eigen12item = ZoomScatterItem()
    eigen12item.sigShowCart.connect(showframe)
    
    eigen13item = ZoomScatterItem()
    eigen13item.sigShowCart.connect(showframe)
    
    eigen23item = ZoomScatterItem()
    eigen23item.sigShowCart.connect(showframe)
    
    eigen14item = ZoomScatterItem()
    eigen14item.sigShowCart.connect(showframe)

    eigen12.addItem(eigen12item)
    eigen13.addItem(eigen13item)
    eigen23.addItem(eigen23item)
    eigen14.addItem(eigen14item)

    roi12 = ExportROI(eigen12item, (0, 0), (1, 1))
    eigen12.addItem(roi12)

    roi13 = ExportROI(eigen13item, (0, 1), (1, 1))
    eigen13.addItem(roi13)

    roi23 = ExportROI(eigen23item, (1, 0), (1, 1))
    eigen23.addItem(roi23)
    
    roi14 = ExportROI(eigen14item, (1, 1), (1, 1))
    eigen14.addItem(roi14)
    
    # Load data
    f = h5py.File(fname,'r')
    saxs = f['data/saxs']
    cart_data = f['data/cart_data']
    ntotal = saxs.shape[0]

    # take a random selection of data
    select_ind = np.unique(np.random.randint(0,ntotal,ndata))

    t0 = time.time()
    if feature == 'saxs':
        d = saxs[select_ind,20:80]
    else:
        d = cart_data[select_ind,400:700,400:600]
        d = d/(np.sum(np.sum(d,axis=-1),axis=-1)+np.finfo(np.float32).eps)[:,None,None]
    print('data loaded in %4.5f seconds'%(time.time()-t0))

    # Do clustering
    eigval, eigvec = clustering.data_to_eig(d, sigma_ref=7)
    cluster_labels = clustering.spectral_clustering(eigvec,nclusters)

    color = iter(plt.cm.hsv(np.linspace(0, 1, nclusters)))

    all_spots12 = []
    all_spots13 = []
    all_spots23 = []
    all_spots14 = []
    all_img_ind = []
    
    for i in range(nclusters):
        c = color.next()
        ind = np.where(cluster_labels == i )
        img_ind = select_ind[ind]

        # brush color
        b = np.array([pg.intColor(i,10)]*np.size(ind))
        print i, np.size(ind)

        x=eigvec[ind,-1].flatten()
        y=eigvec[ind,-2].flatten()
        spots12 = [{'x':xi,'y':yi,'data':img_indi,'brush':bi} \
                for xi,yi,img_indi,bi in zip(x,y,img_ind,b)]

        #eigen12item.setData(spots)
        #eigen12item.setImageIndices(img_ind)
        roi12.setPos((x.min(),y.min()))
        roi12.setSize(((x.max()-x.min())/10.,(y.max()-y.min())/10.))
        
        x=eigvec[ind,-1].flatten()
        y=eigvec[ind,-3].flatten()
        spots13 = [{'x':xi,'y':yi,'data':img_indi,'brush':bi} for xi,yi,img_indi,bi in zip(x,y,img_ind,b)]
        #eigen13item.setData(spots)
        #eigen13item.setImageIndices(img_ind)
        roi13.setPos((x.min(),y.min()))
        roi13.setSize(( (x.max()-x.min())/10.,(y.max()-y.min())/10.))

        x=eigvec[ind,-2].flatten()
        y=eigvec[ind,-3].flatten()
        spots23 = [{'x':xi,'y':yi,'data':img_indi,'brush':bi} for xi,yi,img_indi,bi in zip(x,y,img_ind,b)]
        #eigen23item.setData(spots)
        #eigen23item.setImageIndices(img_ind)
        roi23.setPos((x.min(),y.min()))
        roi23.setSize(( (x.max()-x.min())/10.,(y.max()-y.min())/10.))

        x=eigvec[ind,-1].flatten()
        y=eigvec[ind,-4].flatten()
        spots14 = [{'x':xi,'y':yi,'data':img_indi,'brush':bi} for xi,yi,img_indi,bi in zip(x,y,img_ind,b)]
        #eigen14item.setData(spots)
        #eigen14item.setImageIndices(img_ind)
        roi14.setPos((x.min(),y.min()))
        roi14.setSize(( (x.max()-x.min())/10.,(y.max()-y.min())/10.))

        all_img_ind.append(img_ind)
        for ii in range(np.size(ind)):
            all_spots12.append(spots12[ii])
            all_spots13.append(spots13[ii])
            all_spots23.append(spots23[ii])
            all_spots14.append(spots14[ii])

    eigen12item.setData(all_spots12)
    eigen12item.setImageIndices(all_img_ind)
    
    eigen13item.setData(all_spots13)
    eigen13item.setImageIndices(all_img_ind)

    eigen23item.setData(all_spots23)
    eigen23item.setImageIndices(all_img_ind)

    eigen14item.setData(all_spots14)
    eigen14item.setImageIndices(all_img_ind)

    mainwindow.show()

    QApplication.instance().exec_()
