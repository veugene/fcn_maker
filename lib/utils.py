from __future__ import (print_function,
                        division)
import sys
import os
import shutil



def prepare_training_directory(results_dir):
    PY2 = sys.version_info[0]
    
    if os.path.exists(results_dir):
        print("")
        print("WARNING! Results directory exists: \"{}\"".format(results_dir))
        write_into = None
        while write_into not in ['y', 'n', 'r', '']:
            if PY2:
                write_into = str.lower(raw_input( \
                                 "Write into existing directory? [y/N/r(eplace)]"))
            else:
                write_into = str.lower(input( \
                             "Write into existing directory? [y/N/r(eplace)]"))
        if write_into in ['n', '']:
            print("Aborted")
            sys.exit()
        if write_into=='r':
            shutil.rmtree(results_dir)
            print("WARNING: Deleting existing results directory.")
        print("")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
