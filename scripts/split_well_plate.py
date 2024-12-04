#!/usr/bin/env python
# This script splits a multi well-plate acquisition into several ones
import pycromanager
import argparse
from mmpyreader import mmpyreader
import os 
import shutil
DEBUG = False
#TODO example of usage 

parser = argparse.ArgumentParser(description='''This program splits multi well plates acquired with 
                                                micro-manager''',
                                 epilog='Example of usage:')
parser.add_argument('in_folder',help='the location of the micro-manager dataset ')
parser.add_argument('out_folder',help='where you want the resulting stiched files to be')
parser.add_argument('--wells',nargs='+',help='wells to split, defaults to all')
parser.add_argument('--channels',nargs='+',help='channels to process, defaults to all')
parser.add_argument('--timepoints',nargs='+',help='timepoints to process, defaults to all',type=int)
parser.add_argument('--grid',help='make square grid, needs nx,ny,new_grid_width and index of central well',nargs=4,type=int,default=None)
parser.add_argument('--timestep',help='only keep every n timepoints, starting from',type=int,nargs='+')
parser.add_argument('--downscale',help='downscale by a factor of x',type=int,default=-1)
parser.add_argument('--summary',help='print a summary of the experiment',action='store_true')
#parser.add_argument('-v', '--verbose',action='store_true')
args = parser.parse_args()

# Some checks 
assert os.path.isdir(args.in_folder)
if args.grid: raise NotImplementedError()

# For debugging purposes
if DEBUG:
    print(args.timepoints,args.channels)
    exit(1)

if args.downscale > 0:
    print(f"Will downscale the images by a factor {args.downscale}")
with mmpyreader.MMpyreader() as reader:  
    stack = reader.load_folder(args.in_folder)
    if args.summary:
        stack.print_summary()
        exit() 
    if args.timestep:
        assert len(args.timestep) > 1, 'please provide at least 2 arguments'
        if len(args.timestep) == 2:
            t0,step,end = args.timestep+[stack.dims['T'],]
        elif len(args.timestep) == 3:
            t0,step,end = args.timestep
        timepoints = list(range(t0,end,step))
        print(f"Will only keep timepoints {timepoints}")
    else:
        timepoints = args.timepoints

    stack.split_and_save_wells(args.out_folder,
                               split_timepoints=True,
                               C=args.channels,
                               wells=args.wells,
                               T=timepoints,
                               grid=args.grid,
                               ignore_err=True,
                               scale=args.downscale)
    # copy comments
    commentfile = os.path.join(args.in_folder,'comments.txt')
    shutil.copy(commentfile,os.path.join(args.out_folder,'comments.txt'))