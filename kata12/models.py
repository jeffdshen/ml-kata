import torch
import torch.nn as nn

from kata12.types import Action, Policy, ValueFunction


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding="same")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding="same")

    def forward(self, x: torch.Tensor):
        return x + self.conv2(self.relu(self.conv(x)))


class ConvNet(nn.Module):
    def __init__(
        self,
        input_dim: tuple[int, int],
        hidden_dim: int,
        output_dim: int,
        residual: bool,
    ):
        super().__init__()
        self.net = nn.Sequential(
            *[
                (
                    ResidualBlock(hidden_dim)
                    if residual
                    else nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding="same"),
                        nn.ReLU(),
                    )
                )
                for _ in range(3)
            ]
        )
        self.embed = nn.Embedding(4, hidden_dim)
        self.proj = nn.Linear(hidden_dim * input_dim[0] * input_dim[1], output_dim)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape([*x.shape[:-1], *self.input_dim]).long()
        x = self.embed(x)
        x = x.movedim(-1, -3)
        x = self.net(x)
        x = x.flatten(start_dim=-3)
        return self.proj(x)


class ConvPolicy(Policy):
    def __init__(
        self,
        input_dim: tuple[int, int],
        action_dim: int,
        residual=False,
        hidden_dim=16,
        ffn_dim=32,
    ):
        super().__init__()
        self.net = ConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=ffn_dim,
            residual=residual,
        )
        self.act_proj = nn.Linear(ffn_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_proj(torch.nn.functional.relu(self.net(x)))

    def act(self, observation: list[int]) -> Action:
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32)
            logits: torch.Tensor = self(obs)
            return Action(logits.tolist())


class ConvValue(ValueFunction):
    def __init__(
        self,
        input_dim: tuple[int, int],
        residual: bool = False,
        hidden_dim=16,
        ffn_dim=32,
    ):
        super().__init__()
        self.net = ConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=ffn_dim,
            residual=residual,
        )
        self.value_proj = nn.Linear(ffn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_proj(torch.nn.functional.relu(self.net(x))).squeeze(-1)

    def value(self, observation: list[int]) -> float:
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32)
            return self(obs).item()


class MlpNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.embed = nn.Embedding(4, 4)
        hidden_dim = 128
        # hidden_dim = 8192

        self.net = nn.Sequential(
            nn.Linear(4 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                    for _ in range(2)
                ]
            ),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).flatten(start_dim=-2)
        return self.net(x)


class MlpPolicy(Policy):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = MlpNet(input_dim=input_dim, output_dim=action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.long())

    def act(self, observation: list[int]) -> Action:
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32)
            logits: torch.Tensor = self(obs)
            return Action(logits.tolist())


class MlpValue(ValueFunction):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = MlpNet(input_dim=input_dim, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.long()).squeeze(-1)

    def value(self, observation: list[int]) -> float:
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32)
            return self(obs).item()


class DummyValue(ValueFunction):
    def __init__(self):
        super().__init__()
        self.sink = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = 0.0 * self.sink
        return (zero * x).sum(dim=-1, keepdim=True)

    def value(self, observation: list[int]) -> float:
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32)
            return self(obs).item()
