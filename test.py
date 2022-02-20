import argparse
import torch
import os
import numpy as np
import crowd as crowd
import cv2
import warnings
import utils.log_utils as log_utils
warnings.filterwarnings("ignore")

from two_branch import vgg19


dataset_name='jhu'
score = ''
remarks = ''


parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--model-path', type=str, default=os.path.join('model_save_path/{}/'.format(dataset_name) , score+'.pth'),
                    help='saved model path')
parser.add_argument('--dataset', type=str, default=dataset_name,
                    help='dataset name: qnrf, nwpu, sha, shb')
parser.add_argument('--pred-density-map-path', type=str, 
                    default='output/{}/{}'.format(dataset_name,score),
                    help='save predicted density maps when pred-density-map-path is not empty.')



args = parser.parse_args()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def img_visible(path,img,name):
    mkdir(path)
    vis_img = img[0, 0].cpu().numpy()
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(path,str(name[0]) + '.png'), vis_img)

def new_log(name):
    if os.path.exists (os.path.join(args.pred_density_map_path, name + '.log')):
        os.remove(os.path.join(args.pred_density_map_path , name + '.log'))
    return log_utils.get_logger(os.path.join(args.pred_density_map_path , name + '.log'))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
if args.dataset.lower() == 'qnrf':
    crop_size = 512
    data_path = '/home/zhangy/datasets/QNRF/QNRF-Train-Val-Test'  # UCF-QNRF path
    dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
elif args.dataset.lower() == 'sha' :
    crop_size = 256
    data_path = '/home/zhangy/datasets/Shanghai/part_A_final'  # Shanghai A path
    dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
elif args.dataset.lower() == 'shb':
    crop_size = 512
    data_path = '/home/zhangy/datasets/Shanghai/part_B_final'  # Shanghai B path
    dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
elif args.dataset.lower() == 'jhu':
    crop_size = 512
    data_path = '/home/zhangy/datasets/jhu/Train-Val-Test'  # jhu-crowd++ path
    dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')    
else:
    raise NotImplementedError
dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)


mkdir(args.pred_density_map_path)
model = vgg19().to(device)
model.load_state_dict(torch.load(model_path)) 

model.eval()
image_errs = []
tst = new_log('test')


for inputs,count, name in dataloader:    
    inputs = inputs.to(device)
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    count = count[0].item()

    with torch.set_grad_enabled(False):
        x_map, x_mask= model(inputs)
        
        print(name)
        
        img_visible(os.path.join(args.pred_density_map_path,'mask'),x_mask,name)
        
        img_visible(os.path.join(args.pred_density_map_path,'map'),x_map,name)
        img_visible(os.path.join(args.pred_density_map_path,'f'),x_map*x_map,name)
       
        o = x_mask*x_map+x_map
        img_visible(os.path.join(args.pred_density_map_path,'den'),o,name)


        prediction = torch.sum(o).item()
        count_loss = count - prediction    

        #-----------------------------------------------------------------------

        

      
        
        tst.info('{} with shape = ({} , {}), count loss = {}, gt ={}, prediction = {}'.format(name,inputs.size(2),inputs.size(3),count_loss,count,prediction))


        image_errs.append(count_loss)


mse = np.sqrt(np.mean(np.square(np.array(image_errs))))
mae = np.mean(np.abs(np.array(image_errs)))


print('{}: mae {:.1f}, mse {:.1f}\n'.format(model_path, mae, mse))
os.rename(os.path.join(args.pred_density_map_path) ,
                        os.path.join('output/{}/{:.1f}_{:.1f}_{}'.format(dataset_name, mae,mse,remarks)))

            
