class ModelBase:  # 기반 클래스 선언 object 클래스를 상속받는다.
    def __init__(self, name, dataset):
        self.name = name  # name = flowers_model
        self.dataset = dataset  # dataset = flower데이터셋
        self.is_training = False  # 학습 여부 플래그, 학습중에만 True 검증이나 평가단계에서 False
        if not hasattr(self, 'rand_std'):
            self.rand_std = 0.030

    def __str__(self):  # 문자열 출력시 생성 방법 정의
        return '{}/{}'.format(self.name, self.dataset)

    def exec_all(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3):  # 메인함수 역할
        self.train(epoch_count, batch_size, learning_rate, report)  # 훈련 -> mlp_model_train
        self.test()  # 테스트 -> mlp_model_test
        if show_cnt > 0:
            self.load_visualize(show_cnt)

    def init_parameters(self, hconfigs):
        pass

    def alloc_layer_param(self, input_shape, hconfig):
        pass

    def alloc_param_pair(self, shape):
        pass

    def train(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
        pass

    def test(self):
        pass

    def load_visualize(self, num):
        pass

    def train_step(self,x ,y):
        pass

    def forward_neuralnet(self, x):
        pass

    def backprop_neuralnet(self, G_output, aux):
        pass

    def forward_layer(self, x, hconfig, pm):
        pass

    def backprop_layer(self, G_y, hconfig, pm, aux):
        pass

    def forward_postproc(self, output, y):
        pass

    def forward_extra_cost(self, y):
        pass

    def backprop_postproc(self, G_loss, aux):
        pass

    def backprop_extra_cost(self, G_loss, aux):
        pass

    def eval_accuracy(self, x, y, output=None):
        pass

    def get_estimate(self, x):
        pass