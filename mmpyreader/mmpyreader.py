from pycromanager import Bridge
from itertools import product
import numpy as np
from os import path
from glob import glob
from numpy import array
import subprocess
from tqdm.auto import tqdm
import os
debug = False
import warnings
class MMstack():
    # convert MM channel names to short notation
    __ch_pretty = dict(channel='C',position='P',time='T',z='Z') # we choose this order
    # the reverse mapping
    __ch_mm = dict(zip(__ch_pretty.values(),__ch_pretty.keys()))

    @classmethod
    def _set_bridge(cls,bridge):
        cls.bridge = bridge

    @classmethod
    def _from_store(cls,store):
        '''Create a new stack backed by a store'''
        raise NotImplemented

    def __init__(self,mmstore):
        self._mmstudio = self.bridge.get_studio()
        self._mmstore = mmstore
        self._builder = self._mmstudio.data().getCoordsBuilder()
        self._metadata = self._mmstore.getSummaryMetadata()
        self.channels = self._metadata.getChannelNameList()
        self.positions = self._metadata.getStagePositionList()
        self.positions_names = [p.getLabel() for p in self._metadata.getStagePositionList()]
        self.positions_coords = self._metadata.getStagePositionList()
        self.dims = self._get_dims()

    def get_img(self,P:int = None,T:int = None, C:int = None, Z:int = None,unpack=True,ignore_err=False):
        '''Get image (with metadata!) at specified position, arguments
           order don't matter since those are passed by a dict '''
        coords = dict(P=None,T=None,C=None,Z=None)
        for dim,max in self.dims.items(): # go through all dimensions and check if available
            coords[dim] = self._str_to_idx(dim=dim,val=locals()[dim])
            assert coords[dim] < max ,\
                  f"Required {dim} ({coords[dim]}) exceeds max ({max})"

        # TODO: check that all dimensions exist
        P,T,C,Z = coords.values()
        coords = self._builder.z(Z).time(T).channel(C).stagePosition(P).build()
        img = self._mmstore.getImage(coords)
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

    def _str_to_idx(self,dim = None, val = None):
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


    def _substack_generator(self,P = None,T = None, C = None, Z = None, unpack=True,ignore_err=False):
        ''''''
        coords_prod,_ = self._get_coords_prod(P = P,T = T, C = C, Z = Z)
        h,w = self.get_img(0, 0, 0, 0)[0].shape #assuming all images are of the same size
        n_empty = 0
        # assert len(coords) == np.prod(dims_substack[:-2]),\
        #      f" problem with dimensions {len(coords)} != {np.prod(dims_substack[:-2])}"
        for i,coord in enumerate(coords_prod):
            img = self.get_img(**coord,unpack=unpack,ignore_err=ignore_err)
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
        imgs = list(self._substack_generator(P = P, T = T, C = C, Z = Z, unpack=unpack,ignore_err=ignore_err))
        _ , dims_after = self._get_coords_prod(P = P,T = T, C = C, Z = Z)
        dims_vals = list(dims_after.values())
        if unpack:
            imgs,metadata = [x[0] for x in imgs],[x[1] for x in imgs]
            h,w = imgs[0].shape #assuming all images are of the same size
            imgs = np.array(imgs).reshape(dims_vals+[h,w])
            metadata = np.array(metadata).reshape(dims_vals)
            return imgs, metadata
        else:
            imgs = np.array(imgs).reshape(dims_vals)
            return imgs

    def _get_coords_prod(self,P = None,T = None, C = None, Z = None,new=False,reshape=False):
        ''' Return the coords as a list of dict, or as a ndarray
            new: if the coords are returned as relative coordinates
        '''
        cdict = self._gen_coords(P = P,T = T, C = C, Z = Z)
        if new:
            cdict = {k:list(range(len(v))) for k,v in cdict.items()}
        coords_prod = dict_product(cdict) #cartesian product
        dims = {k:len(v) for k,v in cdict.items()} #list(map(len,cdict.values()))
        if reshape: coords_prod = np.array(coords_prod).reshape(*dims.values())
        return coords_prod, dims

    def get_metadata(self,P = None,T = None, C = None, Z = None,ignore_err=False):
        ''''''
        all_coords,dims_after = self._get_coords_prod(P = P,T = T, C = C, Z = Z)
        dims_vals = list(dims_after.values())
        metadata = []
        for i,coord in tqdm(enumerate(all_coords)):
            img = self.get_img(**coord,unpack=False,ignore_err=ignore_err)
            metadatum = img.getMetadata()
            metadata.append(metadatum)
            img._close()
        metadata = np.array(metadata).reshape(dims_vals)
        return metadata

    def _gen_coords(self,P = None,T = None, C = None, Z = None):
        '''Convert different type of coordinates format to one that is understood by micro-manager'''
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
                coords.append([self._str_to_idx(dim,val)])
            elif type(val) == list:
                if set(list(map(type,val))) == {int} :
                    coords.append(val)
                elif set(list(map(type,val))) == {str} :
                    coords.append(list(map(lambda x: self._str_to_idx(dim,x),val)))
        return dict(zip(self.dims.keys(),coords))

    def _gen_new_coords(self,P = None,T = None, C = None, Z = None):
        ''' return new coordinates as a multi-d array of dict '''
        cdict = self._gen_coords(P = P,T = T, C = C, Z = Z)
        cdict_new = {k:list(range(len(v))) for k,v in cdict.items()}
        return cdict_new

    def save_substack(self,folder,P = None,T = None, C = None, Z = None,repl_img=None,split_pos=False,separate_meta=False,remove_z=False): # same signature as get_substack
        '''Make a real stack object from the coordinates '''
        cdict = self._gen_coords(P = P,T = T, C = C, Z = Z)
        coords_prod = dict_product(cdict) #cartesian product
        coords_prod_new, dims = self._get_coords_prod(new=True, P = P,T = T, C = C, Z = Z)

        Tl,Pl,Zl,Cl = cdict['T'], cdict['P'], cdict['Z'], cdict['C'] #keep list of positions
        store = self._mmstudio.data().createMultipageTIFFDatastore(folder,separate_meta,split_pos)
        smb = self._metadata.copyBuilder()
        ndims = self._builder.z(len(Zl)).time(len(Tl)).channel(len(Cl)).stagePosition(len(Pl)).build()
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
        if repl_img is not None: 
            assert repl_img.shape[:-2] == tuple(dims.values()), f"dims don't match: {repl_img.shape[:-2]} != {tuple(dims.values())}"

        for coord,coord_new in tqdm(zip(coords_prod,coords_prod_new)):
            # old coordinates
            To,Po,Zo,Co  = coord['T'], coord['P'],coord['Z'],coord['C'],
            # new coordinates
            Tn,Pn,Zn,Cn = coord_new['T'],coord_new['P'],coord_new['Z'],coord_new['C']

            if debug: print(f"{[Po,To,Co,Zo]}--->{[Pn,Tn,Cn,Zn]}")
            img = self.get_img(C=Co,P=Po,T=To,Z=Zo,unpack=False)
            mmcordn = self._builder.z(Zn).time(Tn).channel(Cn).stagePosition(Pn).build()
            
            if repl_img is not None: 
                img_np = repl_img[Tn,Pn,Zn,Cn]# get image from input array 
                height,width =  img_np.shape
                bytes_per_pix = img_np.dtype.itemsize
                num_components = 1
                metadata = img.getMetadata()
                acceptable_dtypes = (np.uint16,np.int16,np.uint8,np.float32)
                assert img_np.dtype in acceptable_dtypes, 'cannot process {img_np.dtype}'
                
                if img_np.dtype == np.uint16:
                    # the micro-manager function only takes short or byte arrays
                    img_np = img_np.astype(np.int16)
                elif img_np.dtype == np.float32:
                    # need to create an array which would have the same byte representation
                    # but of the bytes/short type 
                    raise Exception("Micro-manager doesn't work with 32bits arrays")
                
                num_type = img_np.dtype.type
                imgn = self.bridge.get_studio().data().createImage(img_np,
                                                                width,
                                                                height,
                                                                bytes_per_pix,
                                                                num_components,
                                                                mmcordn,
                                                                metadata)
                #FIXME a bit overkill but I really want to be sure that the correct data is saved 
                assert num_type(imgn.getIntensityAt(0,0)) == img_np[0,0], f"{num_type(imgn.getIntensityAt(0,0))} != {img_np[0,0]}"
            
            else:
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
        print("created new susbsack of size: ",end='')
        newstack = type(self)(store)
        store.freeze()
        store.close()
        #I'd want the new stack to be returned but I'm not sure it's the proper way to deal with it
        #return newstack

    def split_and_save_wells(self,root_folder,T = None, C = None, Z = None,wells=None,split_timepoints=False,separate_metadata=True):
        unique_wells = find_unique_wells(self)
        if(len(list(unique_wells))) == len(self.positions):
            print('It doesn\'t seem that your data contains separate wells data')
            return
        
        if wells != None: 
            unique_wells = { well:pos for well,pos in unique_wells.items() if well in wells }

        for well,pos in unique_wells.items():
            out_folder = os.path.join(root_folder,'well_{}'.format(well))
            if split_timepoints: 
                if not T:
                    ts = range(self.dims['T'])
                else:
                    ts = T
                for t in ts:
                    out_folder_t = os.path.join(out_folder,'T{}'.format(t))
                    self.save_substack(P=pos,T = t, C = C, Z = Z,folder=out_folder_t,separate_meta=separate_metadata)
            else: 
                print('{} folder created.'.format(out_folder))
                self.save_substack(P=pos,T = T, C = C, Z = Z,folder=out_folder,separate_meta=separate_metadata)

    def save_substack_fancy(self,folder,P = None,T = None, C = None, Z = None,linkedcoord = None): # same signature as get_substack
        ''' Use some "fancy indexing" so that coordinates of same length are iterated at the same time (zipped)
            Dims will be 4 - # of linked coord + 1 '''
        raise NotImplemented
        if linkedcoord:
            for coord in linkedcoord:
                assert coord in ['P','T','C','Z']

        coords = self._gen_coords(P = P,T = T, C = C, Z = Z)
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
        store = self._mmstudio.data().createMultipageTIFFDatastore(folder,True,True)
        smb = self._mmstudio.data().getSummaryMetadataBuilder()
        # this needs to be changed
        #
        builder = self._builder
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
            builder = self._builder.z(len(Zl)).time(len(Tl)).channel(len(Cl)).stagePosition(len(Pl))

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

    def _get_dims(self,verbose=True): # return all dimensions
        ''' Nicely output dimensions '''
        # Get axes names
        axes = self._metadata.getOrderedAxes()
        axes.reverse()

        # Get number of element
        axnb = list(self._mmstore.getNextIndex(ax) for ax in axes)
        # rename channels with compact versions
        axes = [self.__ch_pretty[ax] for ax in axes]
        ch_dict = dict(zip(axes,axnb))
        # Output the result
        if verbose: print(' x '.join([f"{v}{str(k)}" for k,v in ch_dict.items()]))
        return ch_dict

class MMpyreader():
    def __init__(self,port='auto',verbose=False) -> None:
        # From pycromanager's documentation
        # https://pycro-manager.readthedocs.io/en/latest/mm_java_apis.html
        if port == 'auto':
            import os 
            if os.name == 'nt':
                port =  4827
            elif os.name == 'posix':
                port = os.getuid() + 3827
        
        self.bridge = Bridge(port=port,convert_camel_case=False,iterate=True,debug=False)
        if verbose: print(f"Bridge is opened on port {port}")
        self.mm = self.bridge.get_studio()
        MMstack._set_bridge(self.bridge)

    def load_folder(self,folder: str,virtual: bool = True) -> None:
        if folder.endswith('ome.tif'):
            folder = path.dirname(folder)
        assert path.isdir(folder), f"{folder} doesn't exist"
        assert len(glob(path.join(folder,'*.ome.tif'))) >0, f"No ome.tif file in {folder}"
        mmstore = self.mm.data().loadData(folder,virtual)
        return MMstack(mmstore)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.bridge.close()

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

def dict_product(dic):
    """
    from stackoverflow, takes a dictionary ang generates the cartesian product of its values as dictionnaries
    such that one dict has the keys and the values of one such element of the product
    >>> dict_product(dict(number=[1,2], character='ab'))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return list((dict(zip(dic, e)) for e in product(*dic.values())))

def find_dims_coords(cdict):
    ''' Takes a list of dict and return the associated number of elements for each key '''
    assert type(cdict) == list
    vals = [ ]
    keys = cdict[0].keys()
    merged_dicts =  { k:len(set(e[k] for e in cdict)) for k in keys }
    return merged_dicts

if __name__ == '__main__':
    # Some tests
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    mpl.rc('image', cmap='gray') # set default cmap as gray
    with MMpyreader() as reader:
        stack = reader.load_folder("/mnt/dioscuri-nas/Dioscuri2/Ola/2021.11.19_colonies2/_1",True)
        stack_meta = stack._metadata
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
            print(f"getting image {i}")
            stack.get_img(C=1,P=1,T=i,Z=1)


