from mmengine import Config
baseline_cfg_path = '/mmsegmentation/xuchen_folder/work_dirs/xuchen_mask_segmetation/mask2former_swin-s_8xb2-160k_xuchen-512x512.py'
# /mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_xuchen-512x512.py
cfg = Config.fromfile(baseline_cfg_path)
model_name = baseline_cfg_path.split("/")[-1].split(".")[0]
print(f"config base model of :{model_name} ")

import mmcv
import os
from mmseg.apis import init_model, inference_model, show_result_pyplot
# Init the model from the config and the checkpoint
checkpoint_path = cfg.work_dir +'/best_mIoU_iter_119000.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')

# 推理演示图像
import pylab as plt
import time


data_folder_name ="/mmsegmentation/xuchen_folder/powertain_goods_data"

data_address = data_folder_name +"/test_images"
how_much_file = os.listdir(data_address)
filtered_list = list(filter(lambda how_much_file: how_much_file.endswith(".jpg"), how_much_file))
# model.cfg=cfg
print(f"test_image_address:{data_address}")
print(f"how_much_file:{len(filtered_list)}")
print(f"image_list:{filtered_list} ")


import mmengine
mmengine.mkdir_or_exist(data_folder_name + "/result/")
mmengine.mkdir_or_exist(data_folder_name + "/splited_images_result/")

import time
import matplotlib.pyplot as plt


for i in range(len(filtered_list)):

    image_nn = data_folder_name + "/test_images/" + filtered_list[i][0:-4]
    img_path = image_nn + ".jpg"
    print(f"origin_image_path:{img_path}")
    img = mmcv.imread(img_path)
    resized_img = mmcv.imrescale(img,(512,512))
    print(f"img image size is:{resized_img.shape}")
    # save_resize = image_result_path + "_resize.png"
    # print(f"save_resize_dir:{save_resize}")


        
    start_time = time.time()
    result = inference_model(model, resized_img)
    # print(result)

    
    # Assuming seg_data is your input data
    seg_data = result
    
    # print(result)
    # Extracting fields from the data
    pred_sem_seg_data = seg_data.pred_sem_seg.data.cpu().numpy().squeeze()
    
    end_time = time.time()
    print(f"run time is :{end_time-start_time}")

    image_result_path = data_folder_name + "/result/" + filtered_list[i][0:-4]
    save_resuit_dir = image_result_path + "_result.png"
    print(f"save_resuit_dir:{save_resuit_dir}")
    
    vis_result = show_result_pyplot(model, resized_img,result, draw_gt=False,show=False,out_file=save_resuit_dir)


    # # Displaying the images
    # plt.figure(figsize=(12, 6))
    
    # # Displaying pred_sem_seg image
    # plt.subplot(1, 2, 1)
    # plt.imshow(pred_sem_seg_data) 
    # plt.title('pred_sem_seg')
    
    # # Displaying seg_logits image
    # plt.subplot(1, 2, 2)
    # plt.imshow(resized_img)
    # plt.title('seg_logits')

    # plt.show()
    