import glob
import os
import rasterio

from rasterio.merge import merge
from rasterio.plot import show
from rasterio import features, mask, windows


def sr_to_rgb(root_dir, raster):
    """Writes the analytic scenes to true color (rgb) composites, scaled appropriately with uint8 data type.
    Args:
        root_dir (path): Directory containing the analytic scene(s).
        raster (path): Raster analytic scene filename.
    Returns:
        outfile (path): Raster rgb filepath.
    """
    filename_split = os.path.splitext(raster)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    rgbn = rasterio.open(raster)
    data = rgbn.read([3,2,1])
    scaled = (data * (255 / np.max(data))).astype(np.uint8)
    with rasterio.open(os.path.join(root_dir,f'{basename}_rgb_scaled_uint8.tif'), 'w', driver='GTiff', width=rgbn.width, height=rgbn.height, count=3,crs=rgbn.crs,transform=rgbn.transform,dtype='uint8') as rgb_out:
        rgb_out.write(scaled)
    #plt.imshow(norm.transpose(1,2,0))
    return outfile

def tile(rgb, prefix, width, height, raster_dir, output_dir):
    """Tiles the true color (rgb) composites for use during inference.
    Args:
        rgb (path): Raster rgb filepath.
        prefix (string): Prefix to name tiles with (could be rgb basename).
        width (integer): Tile width.
        height (integer): Tile height.
        raster_dir (path): Raster rgb directory.
        output_dir (path): Tiled raster rgb output directory.
    Returns:
        img_dir (path): Tiles output directory.
    """
    tiles_dir = os.path.join(output_dir,'tiled/')
    img_dir = os.path.join(output_dir,'tiled/rgb/')
    dirs = [tiles_dir, img_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    def get_tiles(ds):
        # get number of rows and columns (pixels) in the entire input image
        nols, nrows = ds.meta['width'], ds.meta['height']
        # get the grid from which tiles will be made 
        offsets = product(range(0, nols, width), range(0, nrows, height))
        # get the window of the entire input image
        big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
        # tile the big window by mini-windows per grid cell
        for col_off, row_off in offsets:
            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
            transform = windows.transform(window, ds.transform)
            yield window, transform

    def crop(inpath, outpath, c):
        # read input image
        image = rasterio.open(inpath)
        # get the metadata 
        meta = image.meta.copy()
        print("meta: ", meta)
        # set the number of channels to 3
        meta['count'] = int(c)
        # set the tile output file format to PNG (saves spatial metadata unlike JPG)
        meta['driver']='PNG'
        meta['dtype']='uint8'
        # tile the input image by the mini-windows
        #i = 0
        for window, transform in enumerate(get_tiles(image)): #get_tiles(image):
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outfile = os.path.join(outpath,"tile_%s_%s.png" % (prefix, str(i)))
            with rasterio.open(outfile, 'w', **meta) as outds:
                imw = image.read(window=window)
                imw = imw.transpose(1,2,0)
                imw = imw[:,:,:3] #rgba2rgb(imw)
                imw = imw.transpose(2,0,1)
                outds.write(imw)
            #i = i+1

    def process_tiles():
        inpath = os.path.join(rgb)
        outpath=img_dir
        crop(inpath, outpath, 3)

    process_tiles()
    return img_dir


def stack_tile(root_dir, raster):
    """Processes the above sr_to_rgb and tiling functions.
    Args:
        root_dir (path): Directory containing the analytic scene(s).
        raster (path): Raster analytic scene filename.
    Returns:
        img_dir (path): Tiles output directory.
    """
    raster_rgb = sr_to_rgb(root_dir, raster)
    filename_split = os.path.splitext(raster_rgb)
    filename_zero, fileext = filename_split
    basename = os.path.basename(filename_zero)
    raster_out_dir = os.path.join(root_dir,basename)
    if not os.path.exists(raster_out_dir):
        os.makedirs(raster_out_dir)

    img_dir = tile(basename+'.tif', str(basename), 96, 96, root_dir, raster_out_dir)
    return img_dir
