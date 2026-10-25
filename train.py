
import torch
from torch import nn

from model import GestureNet


def main() -> None:
    model = GestureNet()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Generate synthetic training data
    inputs = torch.rand(100, 63)
    first_features = inputs[:, 0]
    conditions = first_features > 0.5
    targets = conditions.long()

    # Train the neural network
    model.train()

    for epoch in range(1, 1001):
        optimizer.zero_grad()

        output = model(inputs)
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                predictions = torch.argmax(model(inputs), dim=1)
                accuracy = (predictions == targets).float().mean() * 100

            print(
                f"Epoch {epoch}: Loss = {loss.item():.4f}, "
                f"Accuracy = {accuracy.item():.2f}%"
            )

    # Evaluate on the training data
    model.eval()

    with torch.no_grad():
        predictions = torch.argmax(model(inputs), dim=1)

        print("\nTraining Results:")
        print("Target counts:", torch.bincount(targets, minlength=3))
        print("Prediction counts:", torch.bincount(predictions, minlength=3))

    # Generate unseen test data
    test_inputs = torch.rand(100, 63)
    test_targets = (test_inputs[:, 0] > 0.5).long()

    # Evaluate on the test data
    with torch.no_grad():
        test_outputs = model(test_inputs)
        test_predictions = torch.argmax(test_outputs, dim=1)

        test_accuracy = (
            (test_predictions == test_targets).float().mean() * 100
        )

        print("\nTest Results:")
        print("Target counts:", torch.bincount(test_targets, minlength=3))
        print("Prediction counts:", torch.bincount(test_predictions, minlength=3))
        print(f"Test Accuracy: {test_accuracy.item():.2f}%")


if __name__ == "__main__":
    main()
