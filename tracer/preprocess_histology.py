
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image, ImageEnhance
from scipy import ndimage
import warnings

warnings.simplefilter('ignore', Image.DecompressionBombWarning)


def preprocess_histology(histology_folder):
    """
    Purpose
    -------------
    Pre-process the histological images, resizing, rotating, changing contrast, etc.
    Each image in the folder should be processed separately.
    Currently only .jpg images can be processed and
    
    Inputs
    -------------
    histology_folder : the path of the folder contains the histological images.

    Outputs
    -------------
    The processed images will be saved in the sub-folder 'processed' in histology_folder.

    """
    
    
    if not os.path.exists(histology_folder):
        raise Exception('Folder path %s does not exist. Please give the correct path.' % histology_folder)
    # Directory of histology imagesnew_image = image.resize((400, 400))
    
    # Find the histology files in the folder and save the name
    # i = 0
    # file_name = list()
    # for file in os.listdir(histology_folder):
    #    if fnmatch.fnmatch(file, '*.tif'):
    #        file_name.append(file[0:-4])
    #        i+=1
    file_n = input('Histology file name: ')
    file_name = file_n + '.jpg'
    # Open the histology
    
    histology = Image.open(os.path.join(histology_folder, file_name)).copy()
    # histology = histology.resize((512, 512))
    # histology = resizeimage.resize_crop(histology, [200, 200])
    
    # The modified images will be saved in a subfolder called processed
    path_processed = os.path.join(histology_folder, 'processed')
    if not os.path.exists(path_processed):
        os.mkdir(path_processed)
    
    # Insert the plane of interest
    plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
    # Check if the input is correct
    while plane.lower() != 'c' and plane.lower() != 's' and plane.lower() != 'h':
        print('Error: Wrong plane name')
        plane = input('Select the plane: coronal (c), sagittal (s), or horizontal (h): ')
    
    #  Size in pixels of the reference atlas brain.
    if plane.lower() == 'c':
        atlas_reference_size = (512, 512)
    elif plane.lower() == 's':
        atlas_reference_size = (512, 1024)
    elif plane.lower() == 'h':
        atlas_reference_size = (1024, 512)
    
    if histology.size[0] * histology.size[1] > 89478485:
        oo = histology.size[0] * histology.size[1]
        print('Image size (%d pixels) exceeds limit of 89478485 pixels' % oo)
        print('Image resized to (%d , %d)' % (atlas_reference_size[0], atlas_reference_size[1]))
        histology_width = 1000
        width_percent = (histology_width / float(histology.size[0]))
        height_size = int((float(histology.size[1]) * float(width_percent)))
        histology = histology.resize((histology_width, height_size), Image.NEAREST)
    
    my_dpi = histology.info['dpi'][1]
    pixdim_hist = 25.4 / my_dpi  # 1 inch = 25,4 mm
    # Set up figure
    fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # =============================================================================
    # ax.set_title("Histology viewer")
    # =============================================================================
    # Show the histology image
    ax.imshow(histology)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show(block=True)

    
    if histology.size[0] > atlas_reference_size[0] or histology.size[1] > atlas_reference_size[1]:
        print('\nThe image is ' + str(histology.size[0]) + ' x ' + str(histology.size[1]) + ' pixels')
        print('It is recommended to resize this image down to under ' + str(atlas_reference_size[0]) + ' x ' + str(
            atlas_reference_size[1]) + ' pixels\n')
    
    # Downsample and adjust histology image
    # HistologyBrowser(path_processed,histology)
    print('\nControls: \n')
    print('--------------------------- \n')
    print('a: adjust contrast of the image\n')
    print('r: reset to original\n')
    print('f: continue after contrast adjusting\n')
    print('t: flip figure of 10° anticlockwise\n')
    print('y: flip figure of 10° clockwise\n')
    print('g: Add grid\n')
    print('h: Remove grid\n')
    print('n: set grey scale on\n')
    print('c: crop slice\n')
    print('s: terminate figure editing and save\n')
    print('--------------------------- \n')
    
    print(3)
    # The original istology if needed to be restored
    histology_old = histology.copy()
    F = 0
    histology_copy = histology
    factor = 1
    process_finished = False
    while not process_finished:
        ipt = input('?: ')
        if ipt == 'a':  # if key 'a' is pressed
            print('+ to increase and - to decrease')
            adjusting_finished = False
            while not adjusting_finished:
                adc = input('?contrast: ')
                print(adc)
                enhancer = ImageEnhance.Contrast(histology)
                if adc == '+':
                    factor = factor + 0.1
                    histology = enhancer.enhance(factor)
                    fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi),
                                     dpi=my_dpi)
                    ax = fig.add_subplot(111)
                    # Remove whitespace from around the image
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    # =============================================================================
                    #                 ax.set_title("Slice viewer")
                    # =============================================================================
                    # Show the histology image
                    ax.imshow(histology)
                    plt.tick_params(labelbottom=False, labelleft=False)
                    plt.show(block=True)
                    print('Contrast increased')
                elif adc == '-':
                    factor = factor - 0.1
                    histology = enhancer.enhance(factor)
                    fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi),
                                     dpi=my_dpi)
                    ax = fig.add_subplot(111)
                    # Remove whitespace from around the image
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    # =============================================================================
                    #                 ax.set_title("Slice viewer")
                    # =============================================================================
                    # Show the histology image
                    ax.imshow(histology)
                    plt.tick_params(labelbottom=False, labelleft=False)
                    plt.show(block=True)
                    print('Contrast decreased')
                elif adc == 'r':
                    histology = histology_old
                    factor = 1
                    fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi),
                                     dpi=my_dpi)
                    ax = fig.add_subplot(111)
                    # Remove whitespace from around the image
                    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                    # =============================================================================
                    #                 ax.set_title("Slice viewer")
                    # =============================================================================
                    # Show the histology image
                    ax.imshow(histology)
                    plt.tick_params(labelbottom=False, labelleft=False)
                    plt.show(block=True)
                    print('Original histology restored')
                elif adc == 'f':
                    print('Continue editing')
                    adjusting_finished = True
                    break
        if ipt == 't':  # if key 'q' is pressed
            # histology = histology.rotate(10)
            F += 10
            histology_temp = ndimage.rotate(histology_copy, F, reshape=True)
            histology = Image.fromarray(histology_temp)
            fig = plt.figure(figsize=(float(max(histology_old.size)) / my_dpi, float(max(histology_old.size)) / my_dpi),
                             dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            # Show the histology image
            ax.imshow(histology)
            # ax.margins(x=0, y=-0.25)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('10° rotation')
        elif ipt == 'y':  # if key 'q' is pressed
            # histology = histology.rotate(-10)
            F -= 10
            histology_temp = ndimage.rotate(histology_copy, F, reshape=True)
            histology = Image.fromarray(histology_temp)
            fig = plt.figure(figsize=(float(max(histology.size)) / my_dpi, float(max(histology.size)) / my_dpi),
                             dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            # Show the histology image
            ax.imshow(histology)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('10° rotation')
        elif ipt == 'g':
            fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi), dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            myInterval = 50.
            ax.xaxis.set_major_locator(plticker.IndexLocator(myInterval, 0))
            ax.yaxis.set_major_locator(plticker.IndexLocator(myInterval, 0))
            # Add the grid
            ax.grid(which='major', axis='both', linestyle='-')
            # Add the image
            ax.imshow(histology)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('Gridd added')
        elif ipt == 'h':
            fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi), dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            # Add the grid
            ax.grid(False)
            # Add the image
            ax.imshow(histology)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('Gridd removed')
        elif ipt == 'n':
            histology = histology.convert('LA')
            histology_copy = histology  # for a proper rotation
            fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi), dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            # Show the histology image
            ax.imshow(histology)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('Histology in greyscale')
        elif ipt == 'r':
            F = 0
            histology = histology_old  # restore the original histology
            histology_copy = histology_old  # restore the oiginal histology for rotation
            fig = plt.figure(figsize=(float(histology.size[0]) / my_dpi, float(histology.size[1]) / my_dpi), dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            # Show the histology image
            ax.imshow(histology)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('Original histology restored')
        elif ipt == 'c':
            # Setting the points for cropped image
            left = float(input('Crop horizontal from: '))
            right = float(input('Crop horizontal to: '))
            top = float(input('Crop vertical from: '))
            bottom = float(input('Crop vertical to: '))
            # Cropped image of above dimension
            histology = histology.crop((left, top, right, bottom))
            histology_copy = histology  # for a proper rotation
            fig = plt.figure(figsize=(float(max(histology.size)) / my_dpi, float(max(histology.size)) / my_dpi),
                             dpi=my_dpi)
            ax = fig.add_subplot(111)
            # Remove whitespace from around the image
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # =============================================================================
            #         ax.set_title("Slice viewer")
            # =============================================================================
            # Show the histology image
            ax.imshow(histology)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.show(block=True)
            print('the image is now: ' + str(histology.size[0]) + ' x ' + str(histology.size[1]) + ' pixels')
        elif ipt == 's':
            print(histology)
            histology.save(os.path.join(path_processed, file_n + '_processed.jpg'), 'JPEG', dpi=(my_dpi, my_dpi))
            print('Histology saved')
            process_finished = True
            break






        
    