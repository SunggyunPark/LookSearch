import os
import random
import sys

mode = sys.argv[1]

# Train or Test for generate csv
if mode == 'trn' :
    ims_paths = '/lunit/home/sgpark/data/lookbook/LookBook_Split/train'
    out_paths = '../inputs/looksearch_trn.csv'
    phases = 5
elif mode == 'tst' :
    ims_paths = '/lunit/home/sgpark/data/lookbook/LookBook_Split/test'
    out_paths = '../inputs/looksearch_tst.csv'
    phases = 1
else :
    assert False, 'Type correct mode (ex: trn, tst)'
    
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
with open(  out_paths, 'w' ) as f :

    # Phase means the number of duplicates per one epoch
    for phase in range( phases ) :
        for idx, impath_dirty in enumerate( impaths_dirty ) :
            if idx%1000 == 0 :
                print 'Phase %d -> [%06d/%06d] Generate paired data' %( phase+1, idx, len( impaths_dirty ) )
            
            # Find paired data
            target_pid = pids_dirty[ idx ] 
            impath_clean_pair = impaths_clean[ pids_clean.index( target_pid ) ]
            clean_idx = random.sample( [ idx for idx, pid in enumerate( pids_clean ) if target_pid!=pid ], 1 )
            impath_clean_unpair = impaths_clean[ clean_idx[0] ]
            
            # Check the images paired correctly
            assert int( impath_dirty[3:9] ) == int( impath_clean_pair[3:9] ) 
            assert int( impath_dirty[3:9] ) != int( impath_clean_unpair[3:9] ) 
            
            dirty_caid = int( impath_dirty[31:34] )
            dirty_coid = int( impath_dirty[39:42] )
            clean_pair_caid = int( impath_clean_pair[31:34] )
            clean_pair_coid = int( impath_clean_pair[39:42] )
            clean_unpair_caid = int( impath_clean_unpair[31:34] )
            clean_unpair_coid = int( impath_clean_unpair[39:42] )

            # Write dirty image path, clean paired image path, clean unpaired image path            
            f.write( '%s %s %s ' %( os.path.join( ims_paths, impath_dirty ), os.path.join( ims_paths, impath_clean_pair ), os.path.join( ims_paths, impath_clean_unpair ) ) )
            f.write( '%d %d %d %d %d %d\n' %( dirty_caid, dirty_coid, clean_pair_caid, clean_pair_coid, clean_unpair_caid, clean_unpair_coid ) )
