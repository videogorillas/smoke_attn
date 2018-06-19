
## Spacial

**precision: 0.702740, recall:0.440343; /inception_v3.h5-basic-cropped.mp4.jsonl/result.jsonl**

precision: 0.724624, recall:0.330901; /inception_v3-_288_512_3.h5-basic-cropped.mp4.jsonl/result.jsonl
precision: 0.616613, recall:0.414163; /inception_v3_500_500_3.h5-basic-cropped.mp4.jsonl/result.jsonl

precision: 0.611798, recall:0.275966; /inception_v3_500_500_3.h5-basic.mp4.jsonl/result.jsonl
precision: 0.692205, recall:0.445923; /inception_v3-_288_512_3.h5-basic.mp4.jsonl/result.jsonl
precision: 0.667355, recall:0.415880; /inception_v3.h5-basic.mp4.jsonl/result.jsonl

finetune inception_v3.h5 on vg_smoke; trsh=0.8; 
precision:  0.626943, recall:0.415451 vg_smoke_spacial_v1.1.h5 basic-cropped.mp4

precision: 0.588336, recall:0.147210 vg_smoke_spacial_v1.h5 basic-cropped.mp4

@asvk
precision: 0.669054, recall:0.400858  basic-instinct scenes-detector

thresh=0.5
precision: 0.672103, recall:0.672103 inception_v3-_288_512_3.h5-basic.mp4.jsonl/result.jsonl

## Fusion

train_fusion_v3
Epoch 10/10
19/19 [==============================] - 174s 9s/step - loss: 0.6891 - val_loss: 0.7815


#fusion_vg_smoke_v3.2.h5
thresh 0.8;
precision: 0.599776, recall:0.460515 basic-crop 


#fusion_vg_smoke_mobilenet_v2.4.h5
- normalized opticalflow; 
0.8 precision: 0.718078, recall:0.243777 
0.5 precision: 0.624893, recall:0.624893 (edited)

#i3d_kinetics_finetune_v1.0.hdf; threshold=0.5; basic-cropped.mp4;
precision: 0.611470, recall:0.611470

#i3d_kinetics_finetune_v1.0.hdf; thresh=0.5; 072\ -\ Virginia\ Madsen\ smoking\ style.mp4
precision: 0.585958, recall:0.585958


#i3d_v1.2 
i3d_kinetics_finetune_v1.2.hdf; 0.52; basic-cropped.mp4
precision: 0.464450, recall:0.349288

#i3d_v1.3:
twice more negative samples

#i3d_v1.4:
more trainable params for optical flow model
random crop for clips instead of center (augmentation)

#i3d_kinetics_finetune_v1.5.1
- added activity net positives for training
- added negatives and positives into valset

432/432 [==============================] - 8357s 19s/step - loss: 0.3276 - acc: 0.8663 - val_loss: 0.6445 - val_acc: 0.6944

basic-cropped.mp4
precision: 0.650030, recall:0.469388

#i3d_kinetics_finetune_v1.6.1
- select samples differently
precision: 0.661995, recall:0.533218

#i3d_kinetics_finetune_v1.6.2.hdf; basic-cropped.mp4.jsonl
precision: 0.665658, recall:0.573165

#i3d_kinetics_finetune_v1.7.1.hdf
- new weights='flow_imagenet_and_kinetics'
- train only classifier not FE
