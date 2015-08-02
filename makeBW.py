from PIL import Image
import glob, os
import shutil

def main():
	data_path = "/big1/231n-data/lsp_dataset/"

	if os.path.exists(data_path + "bw"):
		shutil.rmtree(data_path + "bw")

	os.makedirs(data_path + "bw")

	filelist = glob.glob(data_path + "images/*.jpg")	
	for infile in sorted(filelist):
		print infile
		name = os.path.basename(infile)
		print name
		im = Image.open(infile)
		rgb = im.copy()
		out = rgb.convert("L")
		out.save(data_path + "bw/" + name)

if __name__ == "__main__": main()
