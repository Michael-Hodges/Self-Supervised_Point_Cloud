import os
import os.path
import numpy as np
import argparse
import pptk

#class ShapeNetTBD():
#    def __init__(self, path, train_val_test):
#        self.path = path
#        self.pointfiles = os.path.join(self.path, 'train_data')
#
#   def __display__(self,index):
#       x = np.loadtxt(point_data,dtype=np.float32)
#       v = pptk.viewer(x)
#       v.set(point_size-0.001)
#       print("Press enter on visual to continue...")
#       v.wait()
#       v.close()
def display(point_data):
       x = np.loadtxt(point_data,dtype=np.float32)
       v = pptk.viewer(x)
       v.set(point_size=0.001)
       print("Press enter on visual to continue...")
       v.wait()
       v.close()

def display_raw(old_data, point_data):
       x = point_data
       x_prime = old_data
       v = pptk.viewer(x)
       k = pptk.viewer(x_prime)
       v.set(point_size=0.001)
       print("Press enter on visual to continue...")
       k.wait
       v.wait()
       k.close()
       v.close()
def display_voxelized(data, label):
#    for i in range(27):
#        print(i)
    v = pptk.viewer(data,label)
  #  v.attributes(label)
    v.color_map('hsv', scale = [0,26])
    v.set(point_size=0.001)
   
    print("Press enter on visual to continue....")
    v.wait()
    v.close()
        


def rescale_one(point_data):
    x = np.loadtxt(point_data, dtype=np.float32)
#    print( "max: {0}, min:{1}".format(max(x[:,0]), min(x[:,0])))
#    print( "max: {0}, min:{1}".format(max(x[:,1]), min(x[:,1])))
#    print( "max: {0}, min:{1}".format(max(x[:,2]), min(x[:,2])))
#    print(max(x[:,0])-min(x[:,0]))
#    print(max(x[:,1])-min(x[:,1]))
#    print(max(x[:,2])-min(x[:,2]))
#    
#    print(x[0,:].size)
    tmp_max = 0
    for i in range(0,x[0,:].size):
        if max(x[:,i]) - min(x[:,i]) > tmp_max:
                tmp_max = max(x[:,i]) - min(x[:,i])
    scaling_factor = 1/tmp_max    
    new_data = x*scaling_factor
#    print( "max: {0}, min:{1}".format(max(new_data[:,0]), min(new_data[:,0])))
#    print( "max: {0}, min:{1}".format(max(new_data[:,1]), min(new_data[:,1])))
#    print( "max: {0}, min:{1}".format(max(new_data[:,2]), min(new_data[:,2])))
#    print(max(new_data[:,0])-min(new_data[:,0]))
#    print(max(new_data[:,1])-min(new_data[:,1]))
#    print(max(new_data[:,2])-min(new_data[:,2]))
    return x, new_data

def voxelize(x, k):
    # create boundaries
    increment = 1/k
    x_bound = [-0.5]
#    print(x[:,0].size)
    label = np.zeros(x[:,0].size, dtype= np.uint32)
    for i in range(0,k):
        x_bound.append(x_bound[i] + increment)

    bounds = x_bound
#    print(bounds)
    x_bound = x_bound[1:3]
    y_bound = x_bound
    z_bound = x_bound
    voxel_corners=np.array([])
    for i in range(0,k):
        for j in range(0,k):
            for z in range(0,k):
#                print(np.array([bounds[z],bounds[j], bounds[i]]))
                voxel_corners = np.append( voxel_corners, np.array([bounds[z],bounds[j], bounds[i]]))
    voxel_corners = np.reshape(voxel_corners,(k**3,3))

    # apply labels based on voxel
    # there is a much smarter way to do this.
    for i,j in enumerate(x):
        if j[0] < x_bound[0]:
            x_part = 0
        elif j[0] < x_bound[1]:
            x_part = 1
        else: x_part = 2
        if j[1] < y_bound[0]:
            y_part = 0
        elif j[1] < y_bound[1]:
            y_part = 1
        else: y_part = 2
        if j[2] < z_bound[0]:
            z_part = 0
        elif j[2] < z_bound[1]:
            z_part = 1
        else: z_part = 2
        label[i] = x_part + 3*y_part + 9*z_part
          
    x_1 = x
    return x_1, label, voxel_corners

def get_self_supervised_label(point_data, k):
    # scale data to unit cube [-0.5, 0.5]. Currently the data comes in scaled already
    voxel_size = k**3
    x_1 = point_data
    #voxelize over k dimnensions.
    x_1, label, voxel_corners = voxelize(x_1, k)
    #generate new labels by doing random permutation
    rng_gen = np.random.default_rng()
    rand_perm = rng_gen.choice(voxel_size ,size = voxel_size, replace = False)
    #orig_labels = np.range(k^3)
    orig_labels = np.arange(0,voxel_size)
   # perm_label = label
    perm_dict = dict(zip(orig_labels,rand_perm))
#    print(voxel_corners[1])
    #now that we have the dict to map orignal labels to new we need to translate all the points using voxel corners
    # we need to use the dict to create new labels
    perm_label = np.empty(label.size, dtype=np.int32)
    for index, value in enumerate(label):
        perm_label[index] = perm_dict[value]
    # at this poit perm_label has the new labels from the dictionary
    x_translated = np.zeros(shape = x_1.shape)
    for index, value in enumerate(label):
        #old_point + (new_corner-old_corner) gives proper translation
        translation = voxel_corners[perm_label[index]] - voxel_corners[label[index]] 
        x_translated[index] = x_1[index] + translation
#        print(translation)
    # each translation has been applied and the new points are stored in x_translated. We have all we need to do the training
    # the translated points and the orignal labels under the variable label.
#    print('orignal label size: {0}'.format(label.size))
#    print('permuted label size: {0}'.format(perm_label.size))
#    print('x_trans shape: {0}'.format(x_translated.shape))
#    print('translated params')
#    print('x min: {0}, x max: {1}'.format(min(x_translated[:,0]), max(x_translated[:,0])))
#    print('y min: {0}, y max: {1}'.format(min(x_translated[:,1]), max(x_translated[:,1])))
#    print('z min: {0}, z max: {1}'.format(min(x_translated[:,2]), max(x_translated[:,2])))
#    print('original params')
#    print('x min: {0}, x max: {1}'.format(min(x_1[:,0]), max(x_1[:,0])))
#    print('y min: {0}, y max: {1}'.format(min(x_1[:,1]), max(x_1[:,1])))
#    print('z min: {0}, z max: {1}'.format(min(x_1[:,2]), max(x_1[:,2])))

    #uncomment below line if you want to see all of the permutated point voxels
#    display_voxelized(x_translated, label)
    return x_translated, label

def package_point_and_label(points, label):
    np.empty(dtype = np.float32)
    for index, values in enumerate(points):
        print(values)

def convert_shapenet(path):
    #define data paths and create folder for holding new directory if needed
    train_test_val_paths = ['train_data', 'test_data', 'val_data']
    class_paths = ['02691156', '02954340', '03001627', '03467517', '03636649', '03790512', '03948459', '04225987', '02773838', '02958343', '03261776', '03624134', '03642806', '03797390', '04099429', '04379243']
    if not os.path.exists(os.path.join(os.getcwd(), "ShapeNet_Perm")):
        create_path = os.path.join(os.getcwd(), "ShapeNet_Perm")
        for i in train_test_val_paths:
            for j in class_paths:
                os.makedirs(os.path.join(create_path,i,j))
                
    # now that the place to store all data is done we need to iterate through everything and store permutated data
    # read through each data
    print("Permuting data... this will take a while")
    for i in train_test_val_paths:
        print('Permuting {0}'.format(i))
        for j in class_paths:
            data_path = os.path.join(os.getcwd(), path, i , j)
            for item in os.listdir(data_path):
                old_data, scaled_data = rescale_one(os.path.join(data_path, item))
                x_translated, label = get_self_supervised_label(scaled_data,3)
                label_prefix = item.split('.')
                label_name = label_prefix[0] + '_label.txt'
                np.savetxt(os.path.join(os.getcwd(), "ShapeNet_Perm", i, j, item),x_translated, fmt = '%.5f')
                np.savetxt(os.path.join(os.getcwd(), "ShapeNet_Perm", i, j, label_name), label, fmt = '%d')
#                print('path written: {0}'.format( os.path.join(os.getcwd(), "ShapeNet_Perm", i, j, item) ))
#                print(x_translated)
#                display_voxelized(x_translated, label)
    print('done')
#                print(item)
#            print(os.listdir(data_path))







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='ShapeNet', type=str)

    args = parser.parse_args()
    convert_shapenet(args.path)

#    data = ShapeNetTBD(args.path, 'train')
#    display('./ShapeNet_Perm/test_data/02691156/016045.pts')
####    old_data, my_data = rescale_one('./ShapeNet/test_data/02691156/016200.pts')
#    display_raw(old_data, my_data)
 #   x_1, label = voxelize(my_data, 3)
####    x_translated, label = get_self_supervised_label(my_data, 3) 
#    display_voxelized(x_1, label)
