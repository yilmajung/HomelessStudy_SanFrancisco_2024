import json
import os
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict, annotate
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import urllib.request
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import UnidentifiedImageError
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(config):
    print("start_range: ", config["start_range"], "  ",
    "end_range: ", config["end_range"], "  ",
    "box_threshold: ", config["box_threshold"], "  ",
    "text_threshold: ", config["text_threshold"], "  ",
    "text_prompt: ", config["text_prompt"])

    # Set up HOME directory
    HOME = '/gpu02home/wzj5097'

    # #Install GroundingDINO and required packages
    # %cd {HOME}
    # !git clone https://github.com/IDEA-Research/GroundingDINO.git
    # %cd {HOME}/GroundingDINO
    # !pip install -q -e .
    # !pip install -q roboflow
    # !pip install -q supervision


    # Check CONFIG_PATH
    CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

    # %cd {HOME}/GroundingDINO/weights
    # !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


    # Check WEIGHTS_PATH
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO", "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

    # Load Grounding DINO model
    # %cd {HOME}/GroundingDINO
    
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    

    # Create a Retry object
    retry = Retry(total=5, backoff_factor=1)

    # Create a HTTPAdapter with the Retry object
    adapter = HTTPAdapter(max_retries=retry)

    # Create a session and mount the adapter for HTTP and HTTPS
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Create a function for groundingDINO detector

    def grounding_dino_detector(url, model, text_prompt, box_threshold=0.6, text_threshold=0.6):

        '''
        GroundingDINO detector returning boxes (locations of bounding boxes), logits (confidence level) and phrases (label)

        '''

        temp_img = session.get(url, timeout=20).content

        with open('temp_img.jpg', 'wb') as f:
            f.write(temp_img)

        image_source, image = load_image('temp_img.jpg')

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        return boxes, logits, phrases

    # Test if the model works
    url_test = 'https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1240w,f_auto,q_auto:best/rockcms/2023-02/302217-culver-city-homeless-mn-1015-897abc.jpg'
    test = grounding_dino_detector(url_test, model, text_prompt='homeless tent', box_threshold=0.6, text_threshold=0.6)[1]
    print(test)

    start_range = config["start_range"]
    end_range = config["end_range"]
    
    for i in range(start_range, end_range):
        temp_df = pd.read_csv('/gpu02home/wzj5097/GroundingDINO/df_mapillary/df_mapillary_{}.csv'.format(i)) #you should change the path
        temp_df['num_tents'] = 0
        temp_df['confidence'] = 0
        temp_df['confidence'] = temp_df['confidence'].astype('object')

        for j, row in tqdm(temp_df.iterrows()):
            if (j % 100) ==0:
                print(j,' out of ',len(temp_df),' at df_mapillary_',i)

            try:
                temp_results = grounding_dino_detector(temp_df['image_url'][j], model=model,
                                                    text_prompt='tent', box_threshold=0.6, text_threshold=0.6)[1]

                temp_df.loc[j, 'num_tents'] = len(temp_results)
                temp_df.at[j, 'confidence'] = tuple(temp_results.cpu().tolist())

                # Save images include potential tents
                if temp_df['num_tents'][j]>0:
                    urllib.request.urlretrieve(temp_df['image_url'][j], '/gpu02home/wzj5097/GroundingDINO/train_data_img/img_{}_df{}_row{}.jpg'.format(temp_df['image_id'][j],i,j)) #you should change the path

            except requests.exceptions.RequestException as err:
                print(err)

            except requests.exceptions.HTTPError as err:
                print(err)

            except requests.exceptions.ConnectionError as err:
                print(err)

            except requests.exceptions.Timeout as err:
                print(err)

            except UnidentifiedImageError:
                temp_df.loc[i, 'num_tents'] = np.nan
                temp_df.at[i, 'confidence'] = np.nan
                print('UnidentifiedImageError')

            except ValueError:
                temp_df.loc[i, 'num_tents'] = np.nan
                temp_df.at[i, 'confidence'] = np.nan
                print('ValueError')

        if (temp_df['num_tents']>0).any():
            temp_df.to_csv('/gpu02home/wzj5097/GroundingDINO/final_df_yes/df_mapillary_{}.csv'.format(i), index=False) #you should change the path
        
        else:
            temp_df.to_csv('/gpu02home/wzj5097/GroundingDINO/final_df_no/df_mapillary_{}.csv'.format(i), index=False) #you should change the path

if __name__=="__main__":
    with open('config.json', 'r') as file:
        config = json.load(file)

    main(config)
