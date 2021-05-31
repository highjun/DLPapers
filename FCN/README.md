### FCN(Fully Convoluational Network)

Dataset: Pascal VOC 2012  - 2913 segmentation images
Base Network: VGG 16 bn network

#### Model Visualization

#### Experiment
3. FCN-32, batch-size = 4, learning Rate = 2e-5, total Training Time = 21m 
    -  기존 논문에서는 Batch Size =20, lr=1e-4로 설정해서 linear learning rate scaling rule을 적용
4. FCN-16, batch-size = 4, learning Rate = 2e-7, total Training Time = 30m
    - 이전의 FCN-32의 layer 값을 이용해서 Training, drop lr 100times(논문의 방법)
    - 결과가 FCN-32에서 크게 바뀌지 않음
5. FCN-16, batch-size = 4, learning Rate = 2e-5, total Training Time = 30m
    - FCN-32 쓰지 않고 Training
6. FCN-8, batch-size = 4, learning Rate = 2e-5, total Training Time = 30m
7. FCN-8, batch-size = 4, learning Rate = 2e-5, total Training Time = 30m, initialze by FCN16
8. FCN-8, lr = 2e-5, reduceLRonPlateau