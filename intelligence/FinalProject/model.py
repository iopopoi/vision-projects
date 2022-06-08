import torch
from torch import nn


class ModelClass(nn.Module):

    def __init__(self, num_users=610, num_items=193609, rank=10):
        """
        TODO: Write down your model
        """
        super().__init__()
        #1 requires_grad= True : backward
        self.U = torch.nn.Parameter(torch.randn(num_users + 1, rank, requires_grad= True))
        self.V = torch.nn.Parameter(torch.randn(num_items + 1, rank, requires_grad= True))

    def forward(self, users, items):
        ratings = torch.sum(self.U[users] * self.V[items], dim=-1)
        return ratings

class EmbeddingLayers(nn.Module):
    def __init__(self, num_users=610, num_items=193609, rank=10):
        super().__init__()
        self.U = nn.Embedding( num_users + 1 , rank )
        self.V = nn.Embedding( num_items + 1, rank )

        self.U.weight.data.uniform_(0, 0.05)
        self.V.weight.data.uniform_(0, 0.05)

    def forward(self, users, items):
        ratings = torch.sum(self.U(users) * self.V(items), dim=-1)
        return ratings



# class EmbeddingLayers(nn.Module):
#     def __init__(self, num_users=610, num_items=193609, rank=10, factor_num=10):
#         super(EmbeddingLayers, self).__init__()
#         self.factor_num = factor_num

#         # 임베딩 저장공간 확보; num_embeddings, embedding_dim
#         self.embed_user = nn.Embedding(num_users, factor_num)
#         self.embed_item = nn.Embedding(num_items, factor_num)
#         predict_size = factor_num
#         self.predict_layer = torch.ones(predict_size, 1)
#         self._init_weight_()

#     def _init_weight_(self):
#         # weight
#         nn.init.normal_(self.embed_user.weight, std=0.01)
#         nn.init.normal_(self.embed_item.weight, std=0.01)

#         # bias
#         for m in self.modules():
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, users, items):
#         print(users.shape, items.shape)
#         embed_users = self.embed_user(users)
#         embed_items = self.embed_item(items)
#         output_GMF = embed_users * embed_items
#         prediction = torch.matmul(output_GMF, self.predict_layer)
#         return prediction.view(-1)