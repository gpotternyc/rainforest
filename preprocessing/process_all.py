import glob, os, sys

import numpy
import PIL.Image

import process

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python process_all.py [directory]")
		sys.exit(1)
	count = 1
	directory = os.path.join(sys.argv[1], "processed")
	if not os.path.isdir(directory):
		os.mkdir(directory)
	for f in glob.glob(os.path.join(sys.argv[1], "*.jpg")):
		image = numpy.asarray(PIL.Image.open(f))
		processed = PIL.Image.fromarray(process.process(image))
		processed.save(os.path.join(directory, os.path.basename(f)))
		if count % 100 == 0:
			print("Finished {}".format(count))
	print("Done!")
