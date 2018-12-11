import numpy

if __name__ == '__main__':
    basedir = '/blender/storage/home/chexov/macg_embedding_i3v_top/MACG_S02_Ep024_ING_5764188.mov/'
    with open('/tmp/out.tsv', 'w') as _tsv:
        with open('/tmp/l') as _f:
            for l in _f.readlines():
                l = l.strip()
                f = numpy.load(basedir + l)

                s = "\t".join(map(lambda f: str(f), f.tolist()))
                s = s + "\n"
                _tsv.write(s)
