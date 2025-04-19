from .modules import *


class TSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_part = TemporalPart()
        self.spatial_part = SpatialPart()

    def forward(self, x):
        x = self.temporal_part(x)
        x = self.spatial_part(x)
        return x.permute((0, -1, 1))


if __name__ == "__main__":
    from torchinfo import summary
    model = TSModel()
    x = torch.randn(2, 2, 50, 17)
    output = model(x)  # Output shape will be B x 34 x T
    print(output.shape)
    x = torch.randn(2, 2, 100, 17)
    output = model(x)  # Output shape will be B x 34 x T
    print(output.shape)

    summary(model, input_size=(2, 2, 50, 17), col_names=["input_size", "output_size", "num_params", "trainable"], row_settings=["var_names"])