## TSI-CN Dataset Description

### Overview

The TSI-CN dataset is collected using the DJI OSMO Pocket2 camera from the driver's perspective, simulating the position of a traffic recorder to facilitate subsequent algorithm deployment. This dataset includes traffic recording videos from popular cities in China, such as Xi'an, Xianyang, Baoji, Zhengzhou, and Zhoukou. These videos capture typical road scenes, including highways, urban highways, city streets, and rural roads.

### Data Collection Method

During the data collection process, we sampled one key frame every 5 seconds from the recorded videos. These key frames contain various traffic and road scenes, providing rich data support for model training and testing.

### Dataset Structure

The images in the dataset mainly come from the following types of road scenes:
- Highways
- Urban highways
- City streets
- Rural roads

These images cover different types of roads and various weather conditions, lighting changes, and other complex factors encountered in real driving environments.

### Annotation Information

We have conducted detailed annotations on the images in the dataset, primarily focusing on the following aspects:

1. **Guide panel Annotations:**
   - It is divided into seven categories according to the content and basic visual features. They are the prohibit, warning, normal road instruction, highway instruction, scenic area instruction, notice, and dynamic prompt panels, which are labeled with ‘1’∼‘7’ respectively. Meanwhile, considering there are lots of noises (such as architectural graffiti, billboards, etc) that enjoy highly similar visual features with panels, distinguishing them according to traffic-related symbols and texts is an effective and essential way. Therefore, a guide panel will be ignored if all symbols and texts within it are not clearly visible.
   
2. **Symbol Annotations:**
   - It is divide  into two types, including symbola\_a(arrow) and symbolo (warning, instruction, and prohibit), and naming symbol boxes via the combination string of the symbol type and serial number (e.g., ‘w10’ and ‘a2’), where ‘w’ and ‘a’ represent the symbol type and thenumber refers to which particular category it is in a type.

3. **Text Annotations:**
   - The recognition annotation is the corresponding text string, which includes Arabic numerals, English, Chinese, and other special characters. Particularly,considering some strokes of texts stick to each other making it difficult to distinguish them, they are labeled as ‘###’ and ignored
4. **Natural language Annotations:**
   - We follow the Chinese design criteria of road traffic signs to organize the global semantic logic among signs at first. Then, we put the signs that belong to the same semantic logic unit together and describe the traffic instruction information based on the global semantic logic via natural language.


### Data preprocessing
After completing the labeling process with LabelMe, the labels need to go through the following steps to form standardized labels:

**Please note**: the label parts are divided into GT_board and GT_tsa, both of which contain labels for "guide panel", "symbols", "text", "global semantic relationships", and "natural language descriptions", and these need to be integrated together. Be aware that the files `symbol_affiliation.py` and `test_train_index.py` store a dictionary of interpretation contents corresponding to different symbols and the data indexes for the training and testing sets, respectively, which are used to divide the original data into training and test sets.

- **Step 1: Remove Image Information**
Use `del_image_info_from_label_json.py` to manually remove image information from the original labels.Save the processed labels into `GT_board_revision/` and `GT_tsa_revision/` directories.This step ensures that label names and image information align correctly, preventing any labeling confusion.
- **Step 2: Merge Labels**
Execute `board_tsa_merge.py` script to merge board and tsa labels.
Store the merged labels into the `GT_revision/ target` directory.
This process combines the labels for signs ("指示牌") and annotations for symbols, text, overall semantic relationships, and natural language descriptions into a unified format.
- **Step 3: Generate Interpretation Labels for Training and Testing**
Use `language_label_generate_for_test.py` and `language_label_generate_for_train.py` to generate interpretation labels for the testing and training datasets, respectively.
These labels are crucial for understanding and interpreting the symbols, text, semantic relations, etc., in a natural language format.
- **Step 4: Generate Class Indexing**
Run `board_text_symbol_classes_generate.py` to create the `dataset_class.json` file.This file is essential for training and testing processes, facilitating the interpretation of category content through indexes from 0 to n.


#### **If you're eager to dive into the dataset, just hit [this link][data_link] and you're all set !**

[data_link]: https://drive.google.com/file/d/1I2Ae-42iFx0IE5zqUGpI89ZFTfb3T2G_/view?usp=drive_link "Go to TSI-CN Dataset"

