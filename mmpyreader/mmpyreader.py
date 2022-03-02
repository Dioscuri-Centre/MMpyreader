
#%% 
# from importlib.metadata import metadata
from pycromanager import Bridge
from itertools import product
import numpy as np 
from os import path 
from glob import glob
from numpy import array
import subprocess 


class MMstack():
    def __init__(self,mm,stack):
        self.mm = mm
        self.stack = stack
        self.builder = self.mm.data().getCoordsBuilder()
        self.metadata = self.stack.getSummaryMetadata()
        self.channels = self.metadata.getChannelNameList()
        self.positions = self.metadata.getStagePositionList()
        self.positions_names = [p.getLabel() for p in self.metadata.getStagePositionList()]
        self.dims = self.__get_dims()

    def get_img(self,P:int = None,T:int = None, C:int = None, Z:int = None):
        '''Get image (with metadata!) at specified position, arguments 
           order don't matter since those are passed by a dict '''
        for dim in self.dims.keys(): # go through all dimensions and check if available
            assert locals()[dim] < self.dims[dim] ,\
                  f"Required {dim} ({locals()[dim]}) exceeds max ({self.dims[dim]})"
        # TODO: check that all dimensions exist
        coords = self.builder.z(Z).time(T).channel(C).stagePosition(P).build()
        img = self.stack.getImage(coords) 
        assert img != None, 'Could not get image' 
        h,w = img.getHeight(),img.getWidth()
        img_pix = img.getRawPixels().reshape(h,w) #returns a numpy array
        img_metadata = img.getMetadata() 
        return img_pix,img_metadata

    def str_to_idx(self,dim = None, val = None):
        if dim == 'C': 
            assert(val in self.channels), f"{val} not in {self.channels}"
            return self.channels.index(val) 
        if dim == 'P':
            assert(val in self.positions_names), f"{val} not in {self.positions_names}"
            return self.positions_names.index(val) 
        else:
            raise Exception(f'Dimension {dim} cannot be converted to an index')

    def get_substack(self,P = None,T = None, C = None, Z = None,metadata = None):
        ''''''
        coords = [] # we will generate a list of indexes (dicts) to retrieve
        for dim in self.dims.keys(): # go through all dimensions and check if available
            val = locals()[dim]
            if val == None: #if no value, we get the total number of elts
                coords.append(list(range(self.dims[dim])))
            elif type(val) == int: 
                coords.append([val])
            elif type(val) == str: 
                coords.append([self.str_to_idx(dim,val)])    
            elif type(val) == list:
                if set(list(map(type,val))) == {int} :
                    coords.append(val)
                elif set(list(map(type,val))) == {str} :
                    coords.append(list(map(lambda x: self.str_to_idx(dim,x),val)))

        dims_after = list(map(len,coords))
        coords = list(product(*coords)) #cartesian product
        h,w = self.get_img(0, 0, 0, 0)[0].shape #assuming all images are of the same size
        dims_substack = [len(coords),h,w] 
        imgs = np.empty(dims_substack) # array is flattened 
        metadata = []
        assert len(coords) == np.prod(dims_substack[:-2]),\
             f" problem with dimensions {len(coords)} != np.prod(dims_substack[:-2])"
        for i,coord in enumerate(coords):
            coord_dict = {dim:val for dim,val in zip(self.dims.keys(),coord)}
            img, metadatum = self.get_img(**coord_dict)
            imgs[i] = img
            metadata.append(metadatum)    
        imgs = imgs.reshape(dims_after+[h,w]) 
        return imgs,metadata

    def __get_dims(self): # return all dimensions  
        ''' Nicely output dimensions '''
        # Get axes names
        axes = self.metadata.getIntendedDimensions().getAxes()
        # Get number of element
        axnb = list(self.stack.getNextIndex(ax) for ax in axes)
        # rename channels with compact versions
        ch_pretty = dict(channel='C',position='P',time='T',z='Z')
        axes = [ch_pretty[ax] for ax in axes]
        ch_dict = dict(zip(axes,axnb))
        # Output the result 
        print(' x '.join([f"{v}{str(k)}" for k,v in ch_dict.items()]))
        return ch_dict
        
class MMpyreader():
    def __init__(self) -> None:
        # From pycromanager's documentation 
        # https://pycro-manager.readthedocs.io/en/latest/mm_java_apis.html  
        self.bridge = Bridge(convert_camel_case=False,iterate=True,debug=False)
        self.mm = self.bridge.get_studio()

    def load_folder(self,folder: str,virtual: bool = True) -> None:
        assert path.isdir(folder), f"{folder} doesn't exist"
        assert len(glob(path.join(folder,'*.ome.tif'))) >0, f"No ome.tif file in {folder}"
        # self.wine_folder = subprocess.check_output(f"winepath -w {folder}".split(' ')).decode
        # ('UTF-8') # need to convert wine to unix path ... 
        # self.wine_folder = folder.replace('/','Z:/',1)
        stack = self.mm.data().loadData(folder,virtual)
        return MMstack(self.mm,stack)

if __name__ == '__main__':
    # Some tests 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm 
    mpl.rc('image', cmap='gray') # set default cmap as gray
    reader = MMpyreader()
    stack = reader.load_folder("/mnt/dioscuri-nas/Ola/2021.11.19_colonies2/_1",True)
    stack_meta = stack.metadata 
    # Get a single image
    img,img_meta = stack.get_img(C=1,P=1,T=3,Z=1)
    # Get a substack, if an element is not provided it is assumed you want 
    # all elements, /!\ this is 0 indexed
    # 4 elements, channel 0/1 and pos 1/2 
    substack,substack_meta = stack.get_substack(C=[0,1],P=[1,2],T=1,Z=1)
    # Get all zstack 
    substack,substack_meta = stack.get_substack(C=1,P=1,T=1)