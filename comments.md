# Experiments
## 14.02.2024 - 21.02.2024
### Exploratory data analysis
1. Histograms for relative frequency of every attribute in the train, validation and test set, and for the whole dataset.
2. Correlation matrix to see which attributes are correlated
* Most correlated attributes:
| Attribute 1            | Attribute 2            | correlation coefficient|
  | :--------------------- | :--------------------- | :--------------------: |
  | Wearing_Lipstick       | Heavy_Makeup           | 0.8015390              |
  | Smiling                | High_Cheekbones        | 0.6834967              |
  | Smiling                | Mouth_Slightly_Open    | 0.5363789              |
  | Double_Chin            | Chubby                 | 0.5337134              |
  | Sideburns              | Goatee                 | 0.5128934              |
  | Wearing_Lipstick       | Attractive             | 0.4801042              |
  | Heavy_Makeup           | Attractive             | 0.4770836              |
  | Wearing_Lipstick       | Arched_Eyebrows        | 0.4604086              |
  | Mustache               | Goatee                 | 0.4505399              |
  | Heavy_Makeup           | Arched_Eyebrows        | 0.4396449              |
  | Mouth_Slightly_Open    | High_Cheekbones        | 0.4196474              |
  | Wearing_Lipstick       | No_Beard               | 0.4185157              |
  | Male                   | 5_o_Clock_Shadow       | 0.4176698              |

* Least correlated attributes:
| Attribute 1            | Attribute 2            | correlation coefficient|
  | :--------------------- | :--------------------- | :--------------------: |
  | Wearing_Necklace       | Pale_Skin            | 3.754043e-04            |
  | Receding_Hairline      | Black_Hair           | -5.486097e-07           |
  | Rosy_Cheeks            | Narrow_Eyes          | -7.514527e-05           |
  | Sideburns              | Narrow_Eyes          | -1.564910e-04           |
  | High_Cheekbones        | Gray_Hair            | -4.605920e-04           |
  | Mouth_Slightly_Open    | Bald                 | -4.765488e-04           |
  | Black_Hair             | Arched_Eyebrows      | -9.959764e-04           |

* Most negative correlated attributes:
| Attribute 1            | Attribute 2            | correlation coefficient|
  | :--------------------- | :--------------------- | :--------------------: |
  | Male              | Blond Hair          | -0.302988               |
  | Wearing Lipstick  | Big Nose            | -0.303651               |
  | Young             | Double Chin         | -0.309809               |
  | Wavy Hair         | Straight Hair       | -0.321452               |
  | Wavy Hair         | Male                | -0.323983               |
  | Wearing Lipstick  | 5_o_Clock_Shadow    | -0.333921               |
  | Young             | Gray Hair           | -0.364466               |
  | Wearing Earrings  | Male                | -0.373469               |
  | Male              | Attractive          | -0.394451               |
  | Male              | Arched Eyebrows     | -0.408016               |
  | No Beard          | Mustache            | -0.452595               |
  | No Beard          | Male                | -0.522243               |
  | No Beard          | 5_o_Clock_Shadow    | -0.526946               |
  | Sideburns         | No Beard            | -0.543061               |
  | No Beard          | Goatee              | -0.570071               |
  | Male              | Heavy Makeup        | -0.666724               |
  | Wearing Lipstick  | Male                | -0.789435               |

### ROC curves
Zero-shot matching similarity scores are saved in CLIP/EDA/results/result.npy.
Generated ROC curves for every attribute and calculated AUC scores, visualized some AUC distributions per attribute
* Attributes with highest AUC score
| Attribute        | AUC score            |
  | :--------------- | :------------------- |
  | Male             | 0.9900259978199187   |
  | Goatee           | 0.8989014991912996   |
  | Bald             | 0.897193817830388    |
  | Smiling          | 0.8803910493967759   |
  | Blond_Hair       | 0.8772784492577006   |
  | Wearing_Necktie  | 0.8731122087885006   |
  | Eyeglasses       | 0.8540640882587789   |
  | Wearing_Hat      | 0.84516454035606     |
  | Gray_Hair        | 0.8083488725836192   |
  | Bangs            | 0.8077611631914792   |


* Attributes with lowest AUC score
| Attribute        | AUC score            |
  | :--------------- | :------------------- |
  | Arched_Eyebrows    | 0.5979978284282194   |
  | Rosy_Cheeks        | 0.5875902349300913   |
  | Straight_Hair      | 0.5796958544874031   |
  | Young              | 0.5740779819044466   |
  | Big_Nose           | 0.5740005277531861   |
  | High_Cheekbones    | 0.5624101020684489   |
  | Bags_Under_Eyes    | 0.55733985340503     |
  | Pointy_Nose        | 0.528343266806464    |
  | Blurry             | 0.5126279973785057   |
  | Big_Lips           | 0.511572917778934    |
  | Narrow_Eyes        | 0.4707037807079936   |
  | No_Beard           | 0.16473924446936894  |

## 21.02.2024 - 28.02.2024
### ROC curves - continued
1. Paraphrized the captions (CLIP/data/captions/captions_single_attribute_paraphrised.txt) and analysed the results of zero-shot matching
   * Zero-shot matching similarity scores are saved in CLIP/EDA/results/pretrained_clip_paraphrised.npy
2. Zero-shot matching for only negative examples (CLIP/data/captions/captions_single_attribute_negative.txt)
