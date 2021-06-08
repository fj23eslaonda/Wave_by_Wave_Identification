# Wave-by-wave Nearshore Wave Breaking Identificationusing U-Net
A machine learning algorithm based on **the convolutional neural network U-Net** has been trained and validated using as ground truth information a large data set of binary masks obtained from an **automated and independent detection algorithm**.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#data">Data and Methods</a>
    </li>
    <li><a href="#Training">Training</a></li>
    <li><a href="#Validation">Validation</a></li>
    <li><a href="#Results">Results</a></li>
    <li><a href="#Mask">Mask Analysis</a></li>
    <li><a href="#Predicting">Prediction on Las Cruces, Valparaíso, Chile.</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Introduction
 Texto
## Data
Data were collected during a field experiment at the U.S. Army Corp of Engineers Field Research Facility (FRF), Duck, NC. 

<p align="center">
  <img src="Duck images/D252-H10-M00.jpg" alt="Duck Beach, NC, USA." width="300" />
  <img src="Duck images/D252-H12-M00.jpg" alt="Duck Beach, NC, USA." width="300" /> 
</p>


| Model                     | Link                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Train. images (700 frames)**           | [![](https://github.com/caiostringari/deepwaves/blob/master/badges/google_drive_badge.svg)](https://drive.google.com/drive/folders/1HblJpX-V64x5OUzBPa9A6_09cZBBVTDb?usp=sharing) | 
| **Train. masks (700 frames)**           | [![](https://github.com/caiostringari/deepwaves/blob/master/badges/google_drive_badge.svg)](https://drive.google.com/drive/folders/152kT2FAoK257EOAQPUArJC-dBeqxRhaQ?usp=sharing) | 
| **Test images (400 frames)**           | [![](https://github.com/caiostringari/deepwaves/blob/master/badges/google_drive_badge.svg)](https://drive.google.com/drive/folders/1g1GDRPz5IrdpN4PwCK5IfZ-ZYKCHUzpu?usp=sharing) | 
| **Test masks (400 frames)**           | [![](https://github.com/caiostringari/deepwaves/blob/master/badges/google_drive_badge.svg)](https://drive.google.com/drive/folders/1HblJpX-V64x5OUzBPa9A6_09cZBBVTDb?usp=sharing) | 


## Training

## Validation 

|Ranking|F1-Score|Comb. Number |Loss     |Comb. Number|
| ----  | -------|-------------|-----    |----|
|    1  |  0,878 |     11      | 0,00849 | 2 |
|    2  |  0,869 | **3**       | 0,00859 | **27** |
|    3  |  0,863 | 35          | 0,00859 | **3** |
|    4  |  0,858 | **27**      | 0,00862 | **15**|
|    5  |  0,835 | 26          | 0,00871 | 50|
|    6  |  0,821 | 59          | 0,00873 | 7 |
|    7  |  0,811 | **15**      | 0,00878 |**31**|
|    8  |  0,801 | **31**      | 0,00878 | **11**|
|    9  |  0,798 | 34          | 0,00905 | 10 |
|    10 |  0,794 | 25          | 0,00908 | 25 |

|Comb.  | Batch Size| Epochs | Learn.| Drop. | N. Filter| Loss Train. | Loss Val. | F1 Train. | F1 Val.
| ----  |----  |----  |----  |----  |----  |----  |----  |----  |----  |
|**3**      |  **2**  | **50** | **0.010**  | **0.8** | **32** | **0,001** | **0,007** | **0,957**| **0,895** |
| 11    |  2   | 50 | 0.001 | 0.8 | 32 |0,002|  0,008  | 0,929|0,862|
| 15    | 2    | 50 | 0.001 | 0.5 |32  |  0,004 | 0,007    | 0,886      | 0,840  |
| 27    |4     | 50 | 0.010 | 0.8 |32   | 0,004 | 0,010 |0,918|0,863 |
| 31    | 4    | 50 | 0.010 | 0.5 |32  | 0,007 | 0,009 | 0,807 | 0,780|

## Results
<p align="center">
  <img src="Fig/FrameA.jpg" alt="Results Frame A" width="700" />
</p>
<p align="center">
  <img src="Fig/learning_curve-1.png" alt="Learning Curve" width="700" />
</p>

## Mask Analysis


## Prediction on Las Cruces, Valparaíso, Chile.

<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)


[product-screenshot]: images/screenshot.png
