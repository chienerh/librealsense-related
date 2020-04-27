# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
# Import h5py for writing h5 file
import h5py

##################################################################################################################################
##     Read bag from file and write to h5 file                                                                                  ##
##     Reference 1: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/read_bag_example.py     ##
##################################################################################################################################


def main():
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-b", "--bag", type=str, help="Path to the bag file", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/A_9.bag")
    parser.add_argument("-d", "--depth_map_path", type=str, help="Path to the output depth map h5 file", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/rs_a_9_depth_map.h5")
    parser.add_argument("-df", "--depth_folder", type=str, help="Path to the output depth image folder", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/rs_a_9_depth_npy/")
    parser.add_argument("-p", "--point_cloud_path", type=str, help="Path to the output point cloud h5 file", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/rs_a_9_point_cloud.h5")
    parser.add_argument("-pf", "--point_cloud_folder", type=str, help="Path to the output point cloud h5 file", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/rs_a_9_point_clouds/")
    parser.add_argument("-if", "--image_folder", type=str, help="Path to the output image folder", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/rs_a_9_test_images/")
    parser.add_argument("-dif", "--depth_image_folder", type=str, help="Path to the output image folder", default="/home/cel/Umich/EECS545/FinalProject/data/RS/A/rs_a_9_depth_images/")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.bag:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.bag)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    # remove existing hdf5 file
    try:
        os.remove(args.depth_map_path)
        os.remove(args.point_cloud_path)
    except:
        print('no previous h5 file')
    
    # creat path for saving images
    try:
        os.mkdir(args.image_folder)
    except:
        print('image folder already exist')
    try:
        os.mkdir(args.depth_folder)
    except:
        print('depth folder already exist')
    try:
        os.mkdir(args.point_cloud_folder)
    except:
        print('point cloud folder already exist')
    try:
        os.mkdir(args.depth_image_folder)
    except:
        print('depth image folder already exist')

    W = 424
    H = 240

    # write our own h5 file from realsense bag
    f_rs_depth = h5py.File(args.depth_map_path, "a")
    depth_map_id = f_rs_depth.create_dataset('id', (1,), maxshape=(None,), dtype='uint8', chunks=(1,))
    depth_map_from_bag = f_rs_depth.create_dataset('data', (1,H,W), maxshape=(None,H,W), dtype='float16', chunks=(1,H,W))
    depth_map_id[0] = 0
    depth_map_from_bag[0, :, :] = np.zeros((H, W))
    
    f_rs_point_cloud = h5py.File(args.point_cloud_path, "a")
    pcd_id = f_rs_point_cloud.create_dataset('id', (1,), maxshape=(None,), dtype='uint8', chunks=(1,))
    pcd_from_bag = f_rs_point_cloud.create_dataset('data', (1,W*H,3), maxshape=(None,W*H,3), dtype='float16', chunks=(1,W*H,3))
    pcd_id[0] = 0
    pcd_from_bag[0, :, :] = np.zeros((W*H, 3))

    # initialize
    index_list = []

    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.bag, repeat_playback=False)

        # Configure the pipeline to stream the depth stream & color stream
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 30)

        # Start streaming from file
        pipeline.start(config)
               
        index = 0

        # Create colorizer object
        colorizer = rs.colorizer()

        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()

            # Get color frame
            color_frame = frames.get_color_frame()

            # save to list
            index_list.append(index)

            # save depth image
            depth_image = np.asanyarray(depth_frame.get_data()) / 1000
            np.save(args.depth_folder+str(index)+".npy", depth_image)

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            depth_color_image_flipped = cv2.flip(depth_color_image, 1)
            cv2.imwrite(args.depth_image_folder+str(index)+".png", depth_color_image_flipped)

            # save color image
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(args.image_folder+str(index)+".png", color_image)

            # save point cloud
            pc = rs.pointcloud()
            points = rs.points()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            coordinates = np.ndarray(buffer=points.get_vertices(), dtype=np.float16, shape=(W*H, 3))
            np.save(args.point_cloud_folder+str(index)+".npy", coordinates)

            index += 1

    except RuntimeError:
        print("No more frames arrived, reached end of BAG file!")

    finally:
        print('total', len(index_list), 'frames read')

        for index in index_list:
            depth_npy = np.load(args.depth_folder+str(index)+".npy")
            point_cloud_npy = np.load(args.point_cloud_folder+str(index)+".npy")

            # save depth map
            if index == 0:
                # depth
                depth_map_from_bag[-1:,:,:] = depth_npy
                # point cloud
                pcd_from_bag[-1:,:,:] = point_cloud_npy
            else:
                # depth
                depth_map_id.resize(depth_map_id.shape[0]+1, axis=0)   
                depth_map_id[-1:] = index
                depth_map_from_bag.resize((depth_map_from_bag.shape[0]+1, depth_map_from_bag.shape[1], depth_map_from_bag.shape[2]))   
                depth_map_from_bag[-1:,:,:] = depth_npy
                # point cloud
                pcd_id.resize(pcd_id.shape[0]+1, axis=0)   
                pcd_id[-1:] = index
                pcd_from_bag.resize((pcd_from_bag.shape[0]+1, pcd_from_bag.shape[1], pcd_from_bag.shape[2]))   
                pcd_from_bag[-1:,:,:] = point_cloud_npy

        print('depth map h5 file saved:', depth_map_from_bag.shape)
        print('point cloud h5 file saved:', pcd_from_bag.shape)
        
        f_rs_depth.close()
        f_rs_point_cloud.close()


if __name__ == '__main__':
    main()
