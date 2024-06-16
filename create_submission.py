from ultralytics import YOLO
import pandas as pd


def cs(img_path: str, checkpoint_path: str): # , prep_model: str
    """
    Create submission.csv file for private dataset

        Parameters:
            img_path (str): path to uploaded image or directory with images
            checkpoint_path (str): path to pretrained model weights
    """
    model = YOLO(model=checkpoint_path)
    result_yolo = model.predict(source=img_path)

    submit_df0 = pd.DataFrame(index=[0],columns=['fname', 'class'])
    submit_df1 = pd.DataFrame(index=[0],columns=['rel_x', 'rel_y','width','height'])
    j=0
    for sample in result_yolo:
        ndef = sample.boxes.shape[0]
        if ndef==0:
            
            submit_df0.loc[j] = [sample.path.split('/')[-1],'']
            submit_df1.loc[j] = ['','','','']
            j += 1
        else:   
            for i in range(ndef):
                submit_df0.loc[j] = [sample.path.split('/')[-1],int(sample.boxes.cls[i].item())]
                submit_df1.loc[j] = sample.boxes.xywh[i].cpu().numpy().tolist()
                j += 1
    df = pd.concat((submit_df0,submit_df1),axis=1)
    df.to_csv('submission.csv',sep=';',header=False,index=False)
    print(df)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', default="./test_dataset",
                        help='path to uploaded image or directory with images')
    parser.add_argument('-c', '--checkpoint_path', default="./defection_detector/model/checkpoints/YOLOv9c_50epochs.pt",
                        help='path to pretrained model weights')
    args = parser.parse_args()
    img_path = args.img_path
    checkpoint_path = args.checkpoint_path
    cs(img_path, checkpoint_path)
