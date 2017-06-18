import glob, os, sys
import multiprocessing

from joblib import delayed, Parallel
import numpy
import PIL.Image

import process

def process_fn(f, directory):
	image = numpy.asarray(PIL.Image.open(f))
	processed = PIL.Image.fromarray(process.process(image))
	processed.save(os.path.join(directory, os.path.basename(f)))

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python process_all.py [directory]")
		sys.exit(1)
	directory = os.path.join(sys.argv[1], "processed")
	if not os.path.isdir(directory):
		os.mkdir(directory)
	Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5)(
		delayed(process_fn)(f, directory)
			for f in glob.glob(os.path.join(sys.argv[1], "*.jpg"))
	)
	print("Done!")
