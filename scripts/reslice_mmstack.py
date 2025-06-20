#!/usr/bin/env python
# This script only keep the given coordinates of a micro-manager
# stack (positions,times,channels,Z)
import pycromanager
import argparse
from mmpyreader import mmpyreader
from mmpyreader import product
import os
import shutil
DEBUG = False
#TODO example of usage 

parser = argparse.ArgumentParser(description="""This program is aimed at reslicing multi-d acquisiton acquired with micro-manager""")


parser.add_argument("in_folder",help="the location of the micro-manager dataset ")
parser.add_argument("out_folder",help="where you want the resulting stiched files to be")
parser.add_argument("--channels",nargs="+",help="channels to keep, defaults to all")
group = parser.add_mutually_exclusive_group()
group.add_argument('--timepoints',nargs='+',help='timepoints to process, defaults to all',type=int)
group.add_argument("--timestep",help="only keep every n timepoints",type=int)
parser.add_argument("--z",nargs="+",help="z positions to keep, defaults to all",type=int)
parser.add_argument("--positions",nargs="+",help="positions to keep, defaults to all",type=int)
parser.add_argument("--downscale",help="downscale by a factor of x",type=int,default=-1)
parser.add_argument("--nosplit_pos",help="don't split positions (default behaviour)",action="store_false",dest="split_pos")
parser.add_argument("--summary",help="print a summary of the experiment",action="store_true")

args = parser.parse_args()

# Some checks 
assert os.path.isdir(args.in_folder)


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
    timepoints = None
    if args.timestep:
        timepoints = list(range(0,stack.dims['T'],args.timestep))
    elif args.timepoints:
        tstart = args.timepoints[0]
        tend = stack.dims['T'] if args.timepoints[1] == -1 else args.timepoints[1]
        timepoints = list(range(tstart,tend))
        print(f"Will only keep timepoints {args.timepoints}")
    else:
        print(f"Will keep all timepoints ({stack.dims['T']})")

    # Estimate memory required
    mem_req = len(stack._get_coords_prod(                       P=args.positions,
                        C=args.channels,
                        T=timepoints,
                        Z=args.z)[0])*stack.get_img(0,0,0,0)[0].nbytes
    
    if args.downscale != -1: 
        mem_req /= args.downscale**2
    print(f"Disk space required to save: {mem_req/(10**9)} GB")

    stack.save_substack(args.out_folder,
                        P=args.positions,
                        C=args.channels,
                        T=timepoints,
                        Z=args.z,
                        split_pos=args.split_pos,
                        scale=args.downscale)
    # copy comments
    commentfile = os.path.join(args.in_folder,'comments.txt')
    shutil.copy(commentfile,os.path.join(args.out_folder,'comments.txt'))
