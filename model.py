import torch
from torch import nn


class GestureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(63, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x


def main() -> None:
    model = GestureNet()

    hand = torch.rand(1, 63)

    output = model(hand)

    print("Input shape: ", hand.shape)
    print("Output shape: ", output.shape)
    print("Predictions: ", output)

    probabilities = torch.softmax(output, dim=1)

    print("Probabilities: ", probabilities)


if __name__ == "__main__":
    main()