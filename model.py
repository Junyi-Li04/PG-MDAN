class GradReverse(torch.autograd.Function):
  """
  Extension of grad reverse layer
  """
  @staticmethod
  def forward(ctx, x, constant):
    ctx.constant = constant
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    grad_output = grad_output.neg() * ctx.constant
    return grad_output, None

  def grad_reverse(x, constant):
    return GradReverse.apply(x, constant)

class Extractor(nn.Module):

  def __init__(self):
    super(Extractor, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = 40, out_channels = 30, kernel_size = 3, stride = 1,  padding = 1)
    self.bn1 = nn.BatchNorm2d(30)
    self.conv2 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3, stride = 1,  padding = 1)
    self.bn2 = nn.BatchNorm2d(30)

  def forward(self, input):

    x = F.relu(self.bn1(self.conv1(input)))
    x = F.relu(self.bn2(self.conv2(x)))
    # x = self.pool1(x)
    x = torch.reshape(x, (-1, 30*6*12))

    return x


class Predictor(nn.Module):

  def __init__(self):
    super(Predictor, self).__init__()
    self.fc1 = nn.Linear(30*6*12, 40*6)

  def forward(self, input):
    pre = F.relu(self.fc1(input))

    pre = pre.reshape(-1,40,6)

    return pre

class Domain_classifier(nn.Module):

  def __init__(self):
    super(Domain_classifier, self).__init__()
    # self.fc1 = nn.Linear(50 * 4 * 4, 100)
    # self.bn1 = nn.BatchNorm1d(100)
    # self.fc2 = nn.Linear(100, 2)
    self.fc1 = nn.Linear(30*6*12, 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, input, constant):
    input = GradReverse.grad_reverse(input, constant)
    # logits = F.relu(self.bn1(self.fc1(input)))
    # logits = F.log_softmax(self.fc2(logits), 1)
    logits = F.relu(self.fc1(input))

    logits = F.log_softmax(self.fc2(logits), dim = 1)

    return logits
