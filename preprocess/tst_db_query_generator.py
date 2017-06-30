import os
import random
import sys


# Generate test db csv file
ims_paths = '/lunit/home/sgpark/data/lookbook/LookBook_Split/test'
tst_db_paths = '../inputs/looksearch_db_tst.csv'
tst_query_paths = '../inputs/looksearch_query_tst.csv'
phases = 1

# Get list of pids and impath
ims = os.listdir( ims_paths )
impaths_clean = []
impaths_dirty = []
pids_clean = []
pids_dirty = []
for im in ims :
    pid = int( im[3:9] )
    is_clean = int( im[15] )
    if is_clean :
        impaths_clean.append( im )
        pids_clean.append( pid )
    else :
        impaths_dirty.append( im )
        pids_dirty.append( pid )

# Generate csv files
with open(  tst_db_paths, 'w' ) as db :
    with open( tst_query_paths, 'w' ) as query :
        # Generate db csv
        for idx, impath_clean in enumerate( impaths_clean ) :
            target_pid = pids_clean[ idx ]
            db.write( '%s %d\n' %( os.path.join( ims_paths, impath_clean ), target_pid ) )
        for idx, impath_dirty in enumerate( impaths_dirty ) :
            target_pid = pids_dirty[ idx ]
            query.write( '%s %d\n' %( os.path.join( ims_paths, impath_dirty ), target_pid ) )
