# 1장 회귀 분석



퍼셉트론은 동물의 신경세포인 뉴런을 흉내 내어 고안한 것입니다.

이들 퍼셉트론은 저마다 가중치 벡터와 편향값을 이용하여 입력 벡터**x**로 부터 출력 벡터 **y**  를 도출합니다. 

예시로 하나의 퍼셉트론은 벡터 $$\mathbf{x}=(x_1,x_2,x_3,x_4)$$ 이 있으면 가중치 벡터 $$\mathbf{w}=(w_1,w_2,w_3,w_4)$$  과 편향값 b_1을 이용하여 아래 식을 도출 할 수 있습니다.

$$
y_1 = x_1w_1+x_2w_2+x_3w_3+x_4w_4 +b_1
$$


 하나의 퍼셉트론에 관한 식을 반복문을 사용하는 것보다 백터를 사용하면 간단하고 빠르게 구할 수 있어집니다.그래서 이를 다시 적으면 다음과 같아집니다.(x[N,M],])
$$
y_1=\mathbf{x}(1,X)\mathbf{w}(X,1) +b_1
$$
이를 단층 퍼셉트론 신경망은 퍼셉트론이 n개 있을 때 하나의 입력벡터 **x**(1,X) 를 통해 출력 y가 n개가 나옵니다. 그와 동시에 가중치 행렬과 편행 벡터가 사용 되어 출력 벡터 **y**(N,Y)는 다음과 같이 표현할 수 있습니다. 
$$
\mathbf{y}(1,N)=\mathbf{x}(1,X)\mathbf{W}(X,N)+\mathbf{b}(N,N)
$$
여기서 더 나아가 입력 백터를 미니배치를 사용하게 되면 입력벡터 **x**와 출력벡터 **y** 는 벡터에서 행렬로 표현할 수 있게 됩니다. 
$$
\mathbf{Y}(M,N)=\mathbf{X}(N,X)\mathbf{W}(X,N)+\mathbf{b}(M,N)
$$

   

이때 저희가 할 전복 데이터로는 (4177,11)를 사용하여 **X**(4177,10)형태의 입력 행렬과 **Y**(4177,1) 형태의 출력행렬 로 나타낼 수 있고 이때 가중치 **W**는 (10,1) 이됩니다.(**X**(4177,10)**W**(10,1) = **Y**(4177,1))



이를 바탕으로 코드를 분석하게되면 먼저 epoch 수 와 미니배치사이즈 리포트수를 먼저 정합니다.

``` python
def abalone_exec(self, epoch_count=10, mb_size=10, report=20):
```

그후 data를 섞은뒤 80% 기준으로 미니배치사이즈를 바탕으로 미니배치갯수(step_count=334)를 정합니다. 이후 train 데이터와 test 데이터로 나누는 지점(test_begin_idx=3340)을 정합니다.

```python
def arrange_data(self, mb_size):
    self.shuffle_map = np.arange(self.data.shape[0])
    np.random.shuffle(self.shuffle_map)
    step_count = int(self.data.shape[0] * 0.8) // mb_size
    self.test_begin_idx = step_count * mb_size
    return step_count
```
이후 test data와 train data 의 입출력을 정합니다. 이때 train_data는 미니배치 크기만큼 나뉩니다.

출력된 데이터의 형태는 test(**X**[837, 10],**Y** [837, 1])  train( **X**[mb_size(10), 10 ], **Y** [mb_size(10), 1] )

```python
def get_test_data(self):
    test_data = self.data[self.shuffle_map[self.test_begin_idx:]]
    return test_data[:, :-self.output_cnt], test_data[:, -self.output_cnt:]
```
```python
def get_train_data(self, mb_size, nth):
    if nth == 0:
        np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
    train_data = self.data[self.shuffle_map[mb_size * nth:mb_size * (nth + 1)]]
    return train_data[:, :-self.output_cnt], train_data[:, -self.output_cnt:]
```
이제 훈련을 시작하게 되면 먼저 순전파로 $\mathbf{Y}=\mathbf{X}\mathbf{W}+\mathbf{b} $ 를 계산합니다. 이때 **X**(train_x[mb_size(10), 10])와 W[10,1],  b 는 주어진 값이고 **Y**(=output[10,1])는 추정 값이 나오게 됩니다.

```python
def forward_neuralnet(self, x):
    output = np.matmul(x, self.weight) + self.bias
    return output, x
```
나온 output과 train_y를 이용하여 평균제곱오차를 사용하여 loss를 구하여 반환합니다. (전체성분수[데이터의갯수])
$$
평균제곱오차 = (추정값 - 정답)^2/전체 성분수
$$

$$
loss = (output - y)^2/n
$$

```python
def forward_postproc(self, output, y):
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff
```
역전파는 G_loss  ${\partial L\over\partial L}= 1.0$ 을 이용하고 이때 ${\partial L\over\partial {w}}$를 구해야합니다.  먼저 평균제곱오차의 역전파 처리를 하게 됩니다.

이때 loss의 손실 기울기 1로부터$loss = (output - y)^2/n$ 를 output으로 편미분을 하게 되면 ${\partial L\over\partial {output}} = {\partial L\over\partial {square}}{\partial {square}\over\partial {diff}}{\partial {diff}\over\partial {output}}$ 으로 구할 수 있습니다.

 ${\partial L\over\partial {square}}={1\over n}$   (g_loss_square,이때 n=10*1), ${\partial {square}\over\partial {diff}}=2diff$  (g_square_diff)     ${\partial {diff}\over\partial {output}}=1$   (g_diff_output)의 손실 기울기를 구합니다.  결국 ${\partial L\over\partial {output}} = {\partial L\over\partial {square}}{\partial {square}\over\partial {diff}}{\partial {diff}\over\partial {output}} = {2diff\over n}$ 와 같아 집니다.



```python
def backprop_postproc(self, G_loss, diff):
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output
```
이후 ${\partial L\over\partial {output}}$(G_output[mb_size(10),1])와 입력행렬**X**[mb_size(10),10]를 이용하여    ${\partial L\over\partial {w}}={X^T}G_{output}$ 와 ${\partial L\over\partial {b}}$ 은 G_output의 합를 구한 후 $w_{i+1} = w_i-\alpha{\partial L\over\partial {w}} $ 와 $b_{i+1} = x_i-\alpha{\partial L\over\partial {b}} $ 를 구하게됩니다.

```python
def backprop_neuralnet(self, G_output, x):
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output) 
    G_b = np.sum(G_output, axis=0)

    self.weight -= self.LEARNING_RATE * G_w
    self.bias -= self.LEARNING_RATE * G_b
```

이과정을 미니배치 수만큼 진행을 하면서 설정한 report에 맞추어 loss와 정확도를 출력하고 모든 epoch이 끝난 후 테스트를 진행하는데 입력 행렬test_x[837,10] 과 w_fin[837,1]을 이용하여  $\mathbf{Y}=\mathbf{X}\mathbf{W}+\mathbf{b} $  output(Y)을 구한후이를 test_y[837,1] 과 비교하여 정확도(정답과 오차의 비율)를 계산합니다.

```python
def run_test(self, x, y):
    output, _ = self.forward_neuralnet(x)
    accuracy = self.eval_accuracy(output, y)
    return accuracy
```