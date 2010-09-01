function [] = flush_io(fd) 
    Octave = exist('OCTAVE_VERSION');
    if (Octave) 
        fflush(fd); 
    end
    return
end