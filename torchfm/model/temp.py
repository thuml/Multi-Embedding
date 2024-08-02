import torch
import torch.nn as nn

class ME_TE(BaseModel):
    def __init__(self, feature_columns, device, num_tasks, tasks,
            target_name, loss_fn, l2_reg_embedding, tower_hidden_units, dropout,batch_norm,**kwargs):
        super(ME_TE, self).__init__(feature_columns, device, num_tasks, tasks, target_name, loss_fn, l2_reg_embedding)
        self.input_dim = self.get_input_dim(feature_columns)
        self.embedding_dict = nn.ModuleList([create_embedding_matrix(feature_columns, device=self.device) for _ in range(num_tasks)])
        if self.dense_feat_num>0:
            self.dense_feature_embedding = nn.ModuleList([nn.Linear(self.dense_feat_num, self.sparse_feat_dim) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([MLP_Layer(input_dim=self.input_dim,
                                    hidden_units=tower_hidden_units,
                                    output_dim = 1,
                                    dropout_rates=dropout,
                                    batch_norm=batch_norm) for i in range(self.num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        self.regularization_weight = []
        self.rep_gate = nn.Parameter(torch.normal(mean=0., std=1e-4, size=(num_tasks, num_tasks)), requires_grad=True)
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
    def predict(self, inputs):
        input_embedding = []
        if self.dense_feat_num>0:
            task_embedding = [self.get_embedding(inputs, self.embedding_dict[i], self.dense_feature_embedding[i]) for i in range(self.num_tasks)]
        else:
            task_embedding = [self.get_embedding(inputs, self.embedding_dict[i]) for i in range(self.num_tasks)]
        rep_gate = self.rep_gate
        for i in range(self.num_tasks):
            task_input = []
            for j in range(self.num_tasks):
                if j != i :
                    task_input.append(task_embedding[j].detach())
                else:
                    task_input.append(task_embedding[j])
            task_input = torch.stack(task_input,dim=1) # (Batchsize, num_tasks, m*D)
            task_gate = rep_gate[i,:].view(1,-1,1) # (1,num_tasks,1)
            task_input = torch.multiply(task_input,task_gate).sum(1) # (B, m*D)
            input_embedding.append(task_input)
        if not self.training:
            self.cache['rep_gate'] = rep_gate
        # tower 
        output = []
        for i in range(self.num_tasks):
            tower_output = self.towers[i](input_embedding[i]) #(Batchsize, 1)
            tower_output = self.out[i](tower_output)
            output.append(tower_output)
        result = torch.cat(output,-1)
        return result