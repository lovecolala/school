CNN的Convolutional layer皆只有一顆
四個Target，五個Class
['大跌', '小跌', '平盤', '小漲', '大漲']
Class的分類方式為固定值/Boundary
{'>0.02', '0～0.02', '0', '-0.02~0', '<-0.02'}
target使用實數
Final pool feature = 361
應該是有修改Sphere Complex的theta

1. input (19*19)
2. convolutional, nFilter = 1, kernel size = 2*2
3. pooling, kernel size = 2*2
4. convolutional, nFilter = 1, kernel size = 2*2
5. pooling, kernel size = 2*2
6. convolutional, nFilter = 1, kernel size = 3*3