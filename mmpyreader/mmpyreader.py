
#%% 
# from importlib.metadata import metadata
from pycromanager import Bridge
from itertools import product
import numpy as np 
from os import path 
from glob import glob
from numpy import array
import subprocess 
from tqdm.auto import tqdm
import os 

def dict_product(dicts):
    """
    from stackoverflow, takes a dictionary ang generates the cartesian product of its values as dictionnaries
    # such that one dict has the keys and the values of one such element of the product
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return list((dict(zip(dicts, x)) for x in product(*dicts.values())))

class MMstack():
    # convert MM channel names to short notation 
    ch_pretty = dict(channel='C',position='P',time='T',z='Z') # we choose this order 
    # the reverse mapping 
    ch_mm = dict(zip(ch_pretty.values(),ch_pretty.keys())) 

    def __init__(self,bridge,stack):
        self.bridge = bridge
        self.mm = bridge.get_studio()
        self.stack = stack
        self.builder = self.mm.data().getCoordsBuilder()
        self.metadata = self.stack.getSummaryMetadata()
        self.channels = self.metadata.getChannelNameList()
        self.positions = self.metadata.getStagePositionList()
        self.positions_names = [p.getLabel() for p in self.metadata.getStagePositionList()]
        self.dims = self.__get_dims()

    def get_img(self,P:int = None,T:int = None, C:int = None, Z:int = None,unpack=True,ignore_err=False):
        '''Get image (with metadata!) at specified position, arguments 
           order don't matter since those are passed by a dict '''
        coords = dict(P=None,T=None,C=None,Z=None)
        for dim,max in self.dims.items(): # go through all dimensions and check if available
            coords[dim] = self.str_to_idx(dim=dim,val=locals()[dim])
            assert coords[dim] < max ,\
                  f"Required {dim} ({coords[dim]}) exceeds max ({max})"

        # TODO: check that all dimensions exist
        P,T,C,Z = coords.values()
        coords = self.builder.z(Z).time(T).channel(C).stagePosition(P).build()
        img = self.stack.getImage(coords)
        if img == None: 
            if ignore_err: return img
            else : raise Exception('Could not get image')
        if not unpack:
            return img
        h,w = img.getHeight(),img.getWidth()
        img_pix = img.getRawPixels().reshape(h,w) #returns a numpy array
        img_metadata = img.getMetadata() 
        img._close()
        return img_pix,img_metadata 

    def str_to_idx(self,dim = None, val = None):
        if type(val) == int: 
            return val
        elif dim == 'C': 
            assert(val in self.channels), f"{val} not in {self.channels}"
            return self.channels.index(val) 
        elif dim == 'P':
            assert(val in self.positions_names), f"{val} not in {self.positions_names}"
            return self.positions_names.index(val) 
        else:
            raise Exception(f'Dimension {dim} cannot be converted to an index')


    def gen_substack(self,P = None,T = None, C = None, Z = None, unpack=True,ignore_err=False):
        ''''''
        coords = self.gen_coords(P = P, T = T, C = C, Z = Z)
        coords_prod = list(product(*coords)) #cartesian product        
        h,w = self.get_img(0, 0, 0, 0)[0].shape #assuming all images are of the same size
        n_empty = 0
        # assert len(coords) == np.prod(dims_substack[:-2]),\
        #      f" problem with dimensions {len(coords)} != {np.prod(dims_substack[:-2])}"
        for i,coord in enumerate(coords_prod):
            coord_dict = dict(zip(self.dims.keys(),coord))
            img = self.get_img(**coord_dict,unpack=unpack,ignore_err=ignore_err)
            if img == None:
                print(f'found empty image at coords {coord}')
                n_empty +=1
                img = (np.zeros((h,w),dtype=np.uint16),None)
            if unpack:
                img,metadatum = img 
                yield img,metadatum
            else: 
                yield img

    def get_substack(self,P = None,T = None, C = None, Z = None, unpack=True,ignore_err=False):
        ''''''
        
        coords = self.gen_coords(P = P, T = T, C = C, Z = Z)
        coords_prod = list(product(*coords)) #cartesian product
        dims_after = list(map(len,coords))
        h,w = self.get_img(0, 0, 0, 0)[0].shape #assuming all images are of the same size
        dims_substack = [np.prod(dims_after),h,w]
        imgs = list(self.gen_substack(P = P, T = T, C = C, Z = Z, unpack=unpack,ignore_err=ignore_err))

        if unpack: 
            imgs,metadata = [x[0] for x in imgs],[x[1] for x in imgs]
            imgs = np.array(imgs).reshape(dims_after+[h,w])
            metadata = np.array(metadata).reshape(dims_after)
            return imgs, metadata
        else: 
            imgs = np.array(imgs).reshape(dims_after)
            return imgs

    def get_substack_coords(self,P = None,T = None, C = None, Z = None, unpack=True,ignore_err=False):
        '''Just to get the coords'''
        coords = self.gen_coords(P = P, T = T, C = C, Z = Z)
        coords_prod = list(product(*coords)) #cartesian product
        dims_after = list(map(len,coords))
        coords_dict = [dict(zip(self.dims.keys(),x)) for x in coords_prod] 
        coords_dict = np.array(coords_dict).reshape(*dims_after)
        return coords_dict

    def get_metadata(self,P = None,T = None, C = None, Z = None,ignore_err=False):
        ''''''
        cdict = self.gen_coords(P = P,T = T, C = C, Z = Z)
        Tl,Pl,Zl,Cl = cdict['T'], cdict['P'], cdict['Z'], cdict['C'] #keep list of positions
        coords_prod = dict_product(cdict)
        # coords = self.gen_coords(P = P, T = T, C = C, Z = Z)
        # coords_prod = list(product(*coords)) #cartesian product
        dims_after = list(map(len,cdict.values()))
        h,w = self.get_img(0, 0, 0, 0)[0].shape #assuming all images are of the same size
        dims_substack = [np.prod(dims_after),h,w] 
        metadata = []
        n_empty = 0
        for i,coord in tqdm(enumerate(coords_prod)):
            coord_dict = dict(zip(self.dims.keys(),coord))
            img = self.get_img(**coord_dict,unpack=False,ignore_err=ignore_err)
            metadatum = img.getMetadata() 
            metadata.append(metadatum)    
            img._close()

        metadata = np.array(metadata).reshape(dims_after)        
        return metadata

    def gen_coords(self,P = None,T = None, C = None, Z = None):
        coords = [] # we will generate a list of indexes (dicts) to retrieve
        for dim in self.dims.keys(): # go through all dimensions and check if available
            val = locals()[dim]
            if type(val) == np.ndarray:
                val = val.tolist()
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
        return dict(zip(self.dims.keys(),coords))

    def save_substack(self,folder,P = None,T = None, C = None, Z = None,split_pos=False,separate_meta=False,remove_z=False): # same signature as get_substack
        '''Make a real stack object from the coordinates '''
        cdict = self.gen_coords(P = P,T = T, C = C, Z = Z)
        Tl,Pl,Zl,Cl = cdict['T'], cdict['P'], cdict['Z'], cdict['C'] #keep list of positions
        coords_prod = dict_product(cdict) #cartesian product
        #coords_prod = list(product(*cdict.values())) 
        
        store = self.mm.data().createMultipageTIFFDatastore(folder,separate_meta,split_pos)
        smb = self.metadata.copyBuilder()
        ndims = self.builder.z(len(Zl)).time(len(Tl)).channel(len(Cl)).stagePosition(len(Pl)).build()
        smb = smb.intendedDimensions(ndims)
        
        bridge = self.bridge 
        # Set positions
        orig_pos = self.positions
        arraylist_pos = bridge.construct_java_object('java.util.ArrayList')
        for p in Pl:
            pos = orig_pos[p]
            if remove_z:
                zpos = pos.get('ZDrive')
                zpos.set1DPosition('ZDrive',0.0)
            arraylist_pos.add(pos)
        smb = smb.stagePositions(arraylist_pos)

        # Set channels
        arraylist_channels = bridge.construct_java_object('java.util.ArrayList')
        for c in Cl:
            arraylist_channels.add(self.channels[c])
        smb = smb.channelNames(arraylist_channels)


        store.setSummaryMetadata(smb.build())
        
        for coord in tqdm(coords_prod):
            # old coordinates
            #cdict = dict(zip(self.dims.keys(),coord))
            To,Po,Zo,Co  = coord['T'], coord['P'],coord['Z'],coord['C'],
            # new coordinates
            Tn,Pn,Zn,Cn = Tl.index(To),Pl.index(Po),Zl.index(Zo),Cl.index(Co)
            #print(f"{[Po,To,Co,Zo]}--->{[Pn,Tn,Cn,Zn]}")
            mmcord = self.builder.z(Zo).time(To).channel(Co).stagePosition(Po).build()
            img = self.get_img(C=Co,P=Po,T=To,Z=Zo,unpack=False)
            #img = self.stack.getImage(mmcord)
            mmcordn = self.builder.z(Zn).time(Tn).channel(Cn).stagePosition(Pn).build()
            imgn = img.copyAtCoords(mmcordn)
            img._close()
            if remove_z:
                meta = imgn.getMetadata()
                meta_new = meta.copy()
                meta_new.zPositionUm(0.0)
                meta_new = meta_new.build()
                imgn2 = imgn.copyWithMetadata(meta_new)
                imgn._close()
                imgn = imgn2
            store.putImage(imgn)
            imgn._close()
        newstack = type(self)(self.bridge,store) 
        store.freeze()
        store.close()
        return newstack 

    def split_and_save_wells(self,root_folder,T = None, C = None, Z = None,):
        unique_wells = find_unique_wells(self)
        if(len(list(unique_wells))) == len(self.positions):
            print('It doesn\'t seem that your data contains separate wells data')
            return
        for well,pos in unique_wells.items():
            out_folder = os.path.join(root_folder,'well_{}'.format(well))
            print('{} folder created.'.format(out_folder))
            self.save_substack(P=pos,T = T, C = C, Z = Z,folder=out_folder)

    def save_substack_fancy(self,folder,P = None,T = None, C = None, Z = None,linkedcoord = None): # same signature as get_substack
        ''' Use some "fancy indexing" so that coordinates of same length are iterated at the same time (zipped)
            Dims will be 4 - # of linked coord + 1 '''
        if linkedcoord: 
            for coord in linkedcoord: 
                assert coord in ['P','T','C','Z'] 

        coords = self.gen_coords(P = P,T = T, C = C, Z = Z)
        #cdict = { cname:coord for cname,coord in zip(list((self.dims).keys()),coords) }
        cdict = dict(zip(self.dims,coords))
        Cl,Pl,Tl,Zl = coords  #keep list of positions
        # we will reduce the dimensionality  
        coords_prod = list(product(*coords)) #cartesian product

        if linkedcoord: 
            # check all coords are OK 
            assert type(linkedcoord) == str
            # check all coords have same len  
            assert all([cdict[cname] == cdict[linkedcoord[0]] for cname in linkedcoord])
            idxs = [list(cdict.keys()).index(cname) for cname in linkedcoord] 
            valid_coords = list(zip( *[cdict[cname] for cname in linkedcoord] ))
            coords_filtered = []
            for coord in coords_prod:
                # /!\ coords are tuples, cannot compare list to tuple
                relevant_coords =  tuple(coord[idx] for idx in idxs)
                if relevant_coords in valid_coords:
                    coords_filtered.append(coord)

            coords_prod = coords_filtered 
        #store = self.mm.data().createRAMDatastore()
        store = self.mm.data().createMultipageTIFFDatastore(folder,True,True)
        smb = self.mm.data().getSummaryMetadataBuilder()
        # this needs to be changed 
        # 
        builder = self.builder
        relevant_axis = []
        if linkedcoord: 
            # We assume axes have some priority, this could be a parameter  
            ax_priority = ['P','T','C','Z'] 
            ax_priority.reverse()
            map_ax_names  = dict( C = 'channel', Z = 'z', P = 'stagePosition', T = 'time')
            for ax in ax_priority:
                if (ax in linkedcoord) and (len(linkedcoord) > 1): 
                    linkedcoord = linkedcoord.replace(ax,'')
                    getattr(builder,map_ax_names[ax])(0)
                else: 
                    print(ax,len(cdict[ax]),"   ")
                    getattr(builder,map_ax_names[ax])(len(cdict[ax]))
                    relevant_axis.append(ax)
        else: 
            builder = self.builder.z(len(Zl)).time(len(Tl)).channel(len(Cl)).stagePosition(len(Pl))

        ndims = builder.build() 
        sm = smb.intendedDimensions(ndims).build();   
        #store.setSummaryMetadata(sm)

        #TODO: need to finish this 
        for coord in coords_prod:
            # old coordinates
            oldcoord = dict(zip(self.dims,coord))
            Co,Po,To,Zo  = coord 
            #mmcord = self.builder.z(Zo).time(To).
            # (Co).stagePosition(Po).build()
            #img = self.stack.getImage(mmcord)
            img = self.get_img(**oldcoord,unpack=False)
            # new coordinates 
            #Pn,Tn,Cn,Zn = Pl.index(Po),Tl.index(To),Cl.index(Co),Zl.index(Zo)
            #mmcord = self.builder
            newcoord = {}
            for ax,val in zip(self.dims.keys(),coord):
                if ax in relevant_axis: 
                    newcoord[ax] = cdict[ax].index(val)
                else: 
                    #builder.removeAxis(self.ch_mm[ax])
                    newcoord[ax] = 0
                getattr(builder,map_ax_names[ax])(newcoord[ax])

            print(f"{oldcoord }--->{ newcoord }") 
            #mmcord.z(Zn).time(Tn).channel(Cn).stagePosition(Pn).build()
            img = img.copyAtCoords(builder.build())
            store.putImage(img)
        store.freeze()
        store.close()
        #return store 

    def __get_dims(self): # return all dimensions  
        ''' Nicely output dimensions '''
        # Get axes names
        axes = self.metadata.getOrderedAxes()
        axes.reverse()

        # Get number of element
        axnb = list(self.stack.getNextIndex(ax) for ax in axes)
        # rename channels with compact versions
        axes = [self.ch_pretty[ax] for ax in axes]
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
        if folder.endswith('ome.tif'): 
            folder = path.dirname(folder)
        assert path.isdir(folder), f"{folder} doesn't exist"
        assert len(glob(path.join(folder,'*.ome.tif'))) >0, f"No ome.tif file in {folder}"
        stack = self.mm.data().loadData(folder,virtual)
        return MMstack(self.bridge,stack)

    def __enter__(self):
        return self 

    def __exit__(self, type, value, traceback):
        self.bridge.close()

if __name__ == '__main__':
    # Some tests 
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm 
    mpl.rc('image', cmap='gray') # set default cmap as gray
    with MMpyreader() as reader:
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
        
        # get multiple images:
        for i in range(50): 
            stack.get_img(C=1,P=1,T=i,Z=1)
            print(i)


def find_unique_wells(stack):
    import re
    import numpy as np 

    regex = '(.*?)-Pos(.*?)_(.*)'
    labs = [re.search(regex,e) for e in stack.positions_names ]
    res = [lab.group(1) for lab in labs]
    # find all wells labels, keeping insertion order
    all_wells = dict.fromkeys(res)
    print('found {} unique wells'.format(len(all_wells.keys())))
    print('list of wells names:'+ ' '.join([str(x) for x in all_wells.keys()]))
    
    for well in all_wells.keys():
        regex_well = '^{}-Pos(.*?)_(.*)'.format(well)
        is_match = [re.search(regex_well,e) for e in stack.positions_names ]
        idxs = np.flatnonzero(is_match)
        all_wells[well] = idxs
    return all_wells

# %%