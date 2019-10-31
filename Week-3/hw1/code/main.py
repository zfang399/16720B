import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage
import network_layers
if __name__ == '__main__':
    num_cores = util.get_num_CPU()

    # Given input image
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"

    # Q1.1.2 - Check filter responses
    path_img = "../data/aquarium/sun_aztvjgubyrgvirup.jpg"
    image = skimage.io.imread(path_img)
    image = image.astype('float')/255
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

    # Q1.2 - Compute Dictionary for K = 200, alpha = 400
    visual_words.compute_dictionary(num_workers=num_cores)

    # Q1.3 - Generate 3 wordmaps for one category (aquarium)
    dictionary = np.load('dictionary.npy')
    image_path_list = ["../data/aquarium/sun_aztvjgubyrgvirup.jpg",
                  "../data/aquarium/sun_aueaalblfowrtvxb.jpg",
                  "../data/aquarium/sun_djtzmipoyykmljju.jpg"]
    for idx, image_path in enumerate(image_path_list):
        img = skimage.io.imread(image_path)
        img = img.astype('float')/255
        wordmap = visual_words.get_visual_words(img,dictionary)
        util.save_wordmap(wordmap, "wordmap_"+str(idx))

    # Q2.5 - Build and evaluate the SPM based BoW
    visual_recog.build_recognition_system(num_workers=num_cores)
    conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    print(conf)
    print(accuracy, np.diag(conf).sum()/conf.sum())

    # Q3.2 - Train and test VGG16 network
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)

    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())

    #Compare outputs between network layers and Pytorch VGG16
    '''
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    deep_recog.test_deep_features(vgg16)
    '''
