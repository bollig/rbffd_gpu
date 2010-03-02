#!/bin/sh
# Suggested by Doxygen website: 
#svn stat -v $1 | sed -n 's/^[ A-Z?\*|!]\{1,15\}/r/;s/ \{1,15\}/\/r/;s/ .*//p'

# My own beautiful and verbose creation
svn info $1 | grep -i "Last" | awk '{print $3" "$4" "$5";\t"}'
