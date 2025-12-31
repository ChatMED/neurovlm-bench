# chatmed

# Dataset statistics 

Images per dataset:
- brain-tumor-mri-dataset (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset): 7023
  - ~~NOT USED - Whole dataset:~~
    ~~- training:~~ 
    ~~- glioma: 1321~~
      ~~- meningioma: 1339~~
      ~~- pituitary tumor: 1457~~ 
      ~~- no tumor: 1595~~
    ~~- testing:~~
      ~~- glioma: 300~~
      ~~- meningioma: 306~~
      ~~- pituitary tumor: 300~~ 
      ~~- no tumor: 405~~
  ~~- NOT USED - Brain Tumor Classification (MRI) (Sartaj Subset)~~
    ~~- training:~~ 
      ~~- glioma: 826~~
      ~~- meningioma: 822~~
      ~~- pituitary tumor: 827~~ 
      ~~- no tumor: 395~~
    ~~- testing:~~
      ~~- glioma: 100~~
      ~~- meningioma: 115~~
      ~~- pituitary tumor: 74~~
      ~~- no tumor: 105~~  
  - Figshare Subset (MRI T1C+):
    - glioma: 1426
    - meningioma: 708
    - pituitary tumor: 930
  - Br35H (axial): Brain Tumor Detection 2020 Subset
    - Tumor: 1500
    - No tumor: 1500    
- brain tumor images-44: 3957 tumor + 522 No tumor (Glioma-all: 1219 - 382 (T1), **465/464** (T1C+), 372 (T2))
  - Astrocytoma (mapped to Glioma): 580 (176 (T1), **233/232** (T1C+), 171(T2))
  - Carcinoma: 251 (66 (T1), 112 (T1C+), 73(T2))
  - Ependymoma (mapped to Glioma): 150 (45 (T1), 48 (T1C+), 57 (T2))
  - Ganglioglioma (mapped to Glioma): 61 (20 (T1), 18 (T1C+), 23 (T2))
  - Germinoma: 100 (27 (T1), 40 (T1C+), 33 (T2))
  - Glioblastoma (mapped to Glioma): 204 (55 (T1), 94 (T1C+), 55 (T2))
  - Granuloma: 78 (30 (T1), 31 (T1C+), 17 (T2))
  - Medulloblastoma: 131 (23 (T1), 67 (T1C+), 41 (T2))
  - Meningioma: 874 (272 (T1), 369 (T1C+), 233 (T2))
  - Neurocytoma: 457 (130 (T1), 223 (T1C+), 104 (T2))
  - Oligodendroglioma (mapped to Glioma): 224 (86 (T1), 72 (T1C+), 66 (T2))
  - Papilloma: 237 (66 (T1), 108 (T1C+), 63 (T2))
  - Schwannoma: 465 (148 (T1), 194 (T1C+), 123 (T2))
  - Tuberculoma: 145 (28 (T1), 84 (T1C+), 33 (T2))
  - No tumor: 522 (251 (T1), 0 (T1C+), 271 (T2))
- brain tumor images-17 (axial): 4448
  - Glioma (Astrocytoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependymoma): 1317 (**459/430** (T1), **512/508** (T1C+), 346 (T2))
  - Meningioma (Low Grade, Atypical, Anaplastic, Transitional): 1299 (345 (T1), 625 (T1C+), 329 (T2))
  - Neurocytoma (Central - Intraventricular, Extraventricular): 542 (169 (T1), 261 (T1C+), 112 (T2))
  - Schwannoma (Acoustic, Vestibular - Trigeminal): 470 (153 (T1), 194 (T1C+), 123 (T2))
  - Other Abnormalities (Abscesses, Cysts, Miscellaneous Encephalopathies): 257 (152 (T1), 48 (T1C+), 57 (T2))
  - Normal: 563 (272 (T1), 0 (T1C+), 291 (T2))
- Multiple Sclerosis: 3427
  - MS Axial: 650
  - MS Saggital: 761
  - Control Axial: 1002
  - Control Saggital: 1014
- Stroke: 6850
  - Ischemia: 1130 images
  - Bleeding: 1093 images
  - No stroke: 4427 images
  - External test: 200
    - Stroke: 70
    - No stroke: 130
- AISD: 4270
  - Ischemia: 4270
    - Remote infarct: 623
    - Clear acute infarct: 2151
    - Blurred acute infarct: 1223
    - Infarct: 273

Unique classes found: 5

Top 10 classes by image count:
- Tumor: 12149 (old 18136)
- Multiple sclerosis: 1411
- Stroke: 6563 (2293+4270)
- Normal: 9158 (old 13958)
- Other types of injuries: 257
