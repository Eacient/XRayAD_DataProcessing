LABELME_DIR=../data/eggs/Images
OUTPUT_DIR=../data/eggs/voc
LABEL_FILE=../data/eggs/labels.txt
python labelme\examples\semantic_segmentation\labelme2voc.py $LABELME_DIR $OUTPUT_DIR --labels $LBAEL_FILE --noviz