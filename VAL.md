
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

thresh 0.8;
precision: 0.599776, recall:0.460515 basic-crop fusion_vg_smoke_v3.2.h5

