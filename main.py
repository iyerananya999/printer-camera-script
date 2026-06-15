import cv2
import numpy as np
import os
from datetime import datetime
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBAlignMode

def main():

    # Folder path where image will be saved
    save_folder = "caputured_photos"
    if not  os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Initialize Camera Pipeline
    pipeline = Pipeline()
    config = Config()

    try:

        # Enable Color and Depth Streams
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        # Keep Color Lens and Depth Lens aligned
        config.set_align_mode(OBAlignMode.SW_MODE)

    except Exception as e:
        print(f"Error configuring camera profiles: {e}")
        return
    
    # Start the camera
    pipeline.start(config)
    print("[SPACEBAR] to take a picture");
    print("[Q] to exit program")

    try:
        while True:

            # Wait for frames to capture synchronized images
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            # Process Color frame
            color_data = np.frombuffer(color_frame.get_data(), dtype = np.uint8)
            color_img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)

            # Process Depth frame
            depth_data = np.frombuffer(depth_frame.get_data(), dtype = np.uint16)
            depth_img = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

            # Create colormap to view Depth stream
            depth_normalized = cv2.convertScaleAbs(depth_img, alpha = 0.05) 
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # Show the live streams
            cv2.imshow("Live Color Feed", color_img)
            cv2.imshow("Live Depth Heatmap", depth_colormap)

            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF #used '& 0xFF' to omit case sensitivity
            
            # If [SPACEBAR] is pressed
            if key == 32:
                # Generate unique filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                color_filename = os.path.join(save_folder, f"rgb_{timestamp}.png")
                depth_filename = os.path.join(save_folder, f"depth_{timestamp}.png")
                
                # Save the color photo
                cv2.imwrite(color_filename, color_img)
                
                # Save the depth photo
                cv2.imwrite(depth_filename, depth_colormap)
                
                print(f"Saved Color: {color_filename}")
                print(f"Saved Depth: {depth_filename}\n")

            # If [q] is pressed, break the loop and close
            elif key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program interrupted.")

    finally:
        # Safely shut down hardware
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera disconnected.")

if __name__ == "__main__":
    main()

