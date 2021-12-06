# MapSlicing

## Split any image with any degree of overlap.
### Divide image of any size into smaller images of fixed size with any degree of overlapping.

#### Based on [Devyanshu slicing code](https://github.com/Devyanshu/image-split-with-overlap)

## Usage

```bash
python split_image_with_overlap.py -f path/to_file

```

### Arguments

| Short version | Full version | Description
| --------- | --------- | ----------- |
| -h | --help | show this help message and exit |
| -f FILEPATH | --file FILEPATH | File path |
| -W WIDTH | --width WIDTH | Width of cropped image |
| -H HEIGHT | --height HEIGHT | Height of cropped image |
| -O OVERLAP | --overlap OVERLAP | Overlap percentage (int value) |
| -o OUTDIR | --outdir OUTDIR | Output file directory. Empty if same path as original photo. |
| -v | --version | show program version |