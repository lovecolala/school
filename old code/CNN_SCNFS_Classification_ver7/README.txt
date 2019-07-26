將CNN以多顆Neural組成
四個Target，兩個Class
target使用為複數
Final pool feature = 361

1. input (19*19)
2. convolutional, nFilter = 3, kernel size = 2*2
3. pooling, kernel size = 2*2
4. convolutional, nFilter = 6, kernel size = 2*2
5. pooling, kernel size = 2*2
6. fully-connected, size = 4 (for SCNFS)