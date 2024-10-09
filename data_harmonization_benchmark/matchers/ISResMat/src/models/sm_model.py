import os.path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from transformers import BertModel


class AgentsLayer(nn.Module):
    def __init__(self, n_cols, repr_dim=512, scale_factor=10, cols_repr_init_std=0.005, agents=None, ):
        super(AgentsLayer, self).__init__()
        self.n_cols = n_cols
        self.repr_dim = repr_dim
        self.scale_factor = scale_factor
        if agents is None:
            initial_agents = torch.zeros(
                self.n_cols,
                self.repr_dim,
                dtype=torch.float
            )

            nn.init.normal_(initial_agents, mean=0, std=cols_repr_init_std)

            # nn.init.xavier_uniform_(initial_agents)

            # nn.init.kaiming_normal_(initial_agents)

            # initial_agents.fill_(0.005)

            # initial_agents = torch.Tensor(self.n_cols, self.repr_dim).uniform_(-0.001, 0.001)
        else:
            initial_agents = agents
        self._agents = Parameter(initial_agents)

    def forward(self, x):

        # no.6
        # x is detached or not will be decided by the input of this function
        simm = torch.matmul(x, self.agents.T)
        ele = (self.scale_factor * (1 - simm)) ** 2
        prob_ele = 1.0 / (1 + ele)
        t_dist = prob_ele / torch.sum(prob_ele, 1).unsqueeze(1)

        return t_dist

    @property
    def agents(self):
        return F.normalize(self._agents, dim=1)


class BertFoMatching(nn.Module):
    def __init__(self, n_cls_src, n_cls_tgt, output_dim, scale_factor, cont_temperature,
                 sk_n_iter, sk_reg_weight, cols_repr_init_std, model_loc=None):
        super(BertFoMatching, self).__init__()

        # configuration = BertConfig()
        # self.bert = BertModel(configuration)
        # self.bert = BertModel.from_pretrained(
        #     'bert-base-uncased' if model_loc is None else os.path.join(model_loc, 'bert-base-uncased'))
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_config = self.bert.config
        classifier_dropout = (
            bert_config.classifier_dropout if bert_config.classifier_dropout is not None else bert_config.hidden_dropout_prob
        )
        self.n_cls_src = n_cls_src
        self.n_cls_tgt = n_cls_tgt
        self.dropout = nn.Dropout(classifier_dropout)
        self.projector = nn.Sequential(
            nn.Linear(bert_config.hidden_size, output_dim),
            # nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.src_agents_layer = AgentsLayer(self.n_cls_src, output_dim, scale_factor,
                                            cols_repr_init_std=cols_repr_init_std)
        self.tgt_agents_layer = AgentsLayer(self.n_cls_tgt, output_dim, scale_factor,
                                            cols_repr_init_std=cols_repr_init_std)

        self.cont_temperature = cont_temperature
        self.sk_n_iter = sk_n_iter
        self.sk_reg_weight = sk_reg_weight

        self.CE_loss_func = torch.nn.CrossEntropyLoss()
        self.KL_loss_func = nn.KLDivLoss(reduction='batchmean')
        self.curr_opt_trans_sim_matrix = None

    def forward(self, cls_token_id, input_ids, attention_mask, token_type_ids, label_ls, data_source,
                return_logits=False, ):
        assert data_source == 'src' or data_source == 'tgt'
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]

        # The id of the [CLS] token in bert is 101.
        # The way embeddings are obtained may seem a bit peculiar, mainly because of
        # early exploration of different fragment construction methods in the project.
        cls_indexes = torch.nonzero(input_ids == cls_token_id)
        cls_logits = torch.zeros(cls_indexes.shape[0],
                                      sequence_output.shape[2]).to(input_ids.device)
        for n in range(cls_indexes.shape[0]):
            i, j = cls_indexes[n]
            logit_n = sequence_output[i, j, :]
            cls_logits[n] = logit_n

        cls_logits = self.dropout(cls_logits)
        orig_cls_logits = self.projector(cls_logits)
        cls_logits = F.normalize(orig_cls_logits, dim=1)

        if not return_logits:
            # Do not change the original label_ls, since it will be used in the computation of the
            # meta-matching loss, the reformation here is for the computation of meta-matching loss
            flatten_label_ls = [sub_lst for lst in label_ls for sub_lst in lst]
            total_labels_length = sum(len(labels) for labels in flatten_label_ls)
            assert total_labels_length == len(cls_logits)

            match_loss = self.compute_batch_cols_contrastive_loss(cls_logits, label_ls)
            self_assign_loss = self.self_assign_dist_loss(cls_logits, flatten_label_ls, data_source)

            return match_loss, self_assign_loss
        else:
            return cls_logits

    def self_assign_dist_loss(self, cls_logits, flatten_label_ls, data_source=None):
        if data_source == 'src':
            self_target_dist = self.self_target_distribution(flatten_label_ls,
                                                             self.n_cls_src)
            self_assign_prob = self.src_agents_layer(cls_logits)
        elif data_source == 'tgt':
            self_target_dist = self.self_target_distribution(flatten_label_ls,
                                                             self.n_cls_tgt)
            self_assign_prob = self.tgt_agents_layer(cls_logits)
        dist_loss = self.KL_loss_func(self_assign_prob.log(), self_target_dist)
        return dist_loss

    # def cross_assign_dist_loss(self, cls_logits, flatten_label_ls, data_source=None):
    #     # when the program just begins
    #     if self.curr_opt_trans_sim_matrix == None:
    #         return torch.tensor(0.)
    #     cross_target_dist = self.cross_target_distribution(flatten_label_ls, data_source)
    #     if data_source == 'src':
    #         cross_assign_prob = self.tgt_agents_layer(cls_logits)
    #     if data_source == 'tgt':
    #         cross_assign_prob = self.src_agents_layer(cls_logits)
    #     cross_assign_loss = self.KL_loss_func(cross_assign_prob.log(), cross_target_dist)
    #     return cross_assign_loss

    def self_target_distribution(self, label_ls, n_cls):
        concatenated = torch.cat(label_ls)
        self_tgt_dist = torch.nn.functional.one_hot(concatenated, num_classes=n_cls).float()
        return self_tgt_dist.float().detach()

    def compute_batch_cols_contrastive_loss(self, logits, labels_ls):
        num_inner_batch = len(labels_ls)
        # for every pairwise fragments, the totol columns it contains is the same
        chunked_logits = logits.view(num_inner_batch, -1, logits.shape[1])
        similarity_matrix = torch.bmm(chunked_logits, torch.transpose(chunked_logits, 1, 2))

        def match_label_builder(sublist):
            conc_label_tensor = torch.cat(sublist)
            match_labels = (conc_label_tensor.unsqueeze(1) == conc_label_tensor.unsqueeze(0)).float()
            return match_labels

        match_label_ls = [match_label_builder(sublist) for sublist in labels_ls]
        match_labels = torch.stack(match_label_ls)

        eye_matrix = ~torch.eye(match_labels.shape[1], dtype=bool).to(logits.device)
        mask = torch.stack([eye_matrix] * num_inner_batch, dim=0)

        similarity_matrix = torch.masked_select(similarity_matrix, mask).view(similarity_matrix.shape[0],
                                                                              similarity_matrix.shape[1], -1)
        match_labels = torch.masked_select(match_labels, mask).view(match_labels.shape[0],
                                                                    match_labels.shape[1], -1)

        similarity_matrix = similarity_matrix.view(-1, similarity_matrix.shape[2])
        match_labels = match_labels.view(-1, match_labels.shape[2])

        # the logits output from the model is already normalized
        local_logits = similarity_matrix / self.cont_temperature
        loss = self.CE_loss_func(local_logits, match_labels)
        return loss

    def get_sim_matrix_with_recloss(self):
        src_agents = self.src_agents_layer.agents
        tgt_agents = self.tgt_agents_layer.agents
        sim_matrix = torch.matmul(src_agents, tgt_agents.T)

        # sim_matrix_loss = ((sim_matrix + 1) ** 2).sum()
        # sim_matrix_loss = sim_matrix.abs() ** 2

        # mutual_sim_matrix_A = sim_matrix / ((sim_matrix + 1).sum(dim=1).unsqueeze(1))
        # mutual_sim_matrix_B = sim_matrix / ((sim_matrix + 1).sum(dim=0))
        # # mutual_sim_matrix = (mutual_sim_matrix_A + mutual_sim_matrix_B) / (sim_matrix.shape[0] + sim_matrix.shape[1])
        # mutual_sim_matrix = (mutual_sim_matrix_A + mutual_sim_matrix_B)

        # sim_matrix_cpu = sim_matrix.detach().cpu().numpy()
        # # sim_matrix_cpu = mutual_sim_matrix.detach().cpu().numpy()
        # ind = linear_sum_assignment(sim_matrix_cpu.max() - sim_matrix_cpu)
        # mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        # mask[ind[0], ind[1]] = False
        # masked_tensor = sim_matrix[mask]
        # sim_matrix_loss = ((masked_tensor + 1) ** 2).sum()

        # flag = torch.ones_like((sim_matrix), dtype=torch.bool).to(
        #     src_agents.device)
        # mask = torch.ones_like(sim_matrix, dtype=torch.bool).to(
        #     src_agents.device)
        # while torch.any(flag):
        #     sim_masked = torch.where(flag, sim_matrix, float('-inf'))
        #     max_pos = (sim_masked == torch.max(sim_masked)).nonzero()
        #     flag[max_pos[0][0], :] = False
        #     flag[:, max_pos[0][1]] = False
        #     mask[max_pos[0][0], max_pos[0][1]] = 0
        # masked_tensor = sim_matrix[mask]
        # # pesudo_match_tensor = sim_matrix[~mask]
        # # sim_matrix_loss = ((masked_tensor + 1) ** 2).sum() + (-((pesudo_match_tensor + 1) ** 2).sum())
        # sim_matrix_loss = ((masked_tensor + 1) ** 2).sum()

        opt_trans_sim_matrix = self.compute_optimal_transport(sim_matrix.detach(), lam=self.sk_reg_weight,
                                                              niter=self.sk_n_iter)

        self.curr_opt_trans_sim_matrix = opt_trans_sim_matrix

        # NOTE: the similarity matrix return here will be used to decide the match based on its values
        return sim_matrix, opt_trans_sim_matrix

    def compute_optimal_transport(cls, sim_matrix, lam=1, niter=5):
        n, m = sim_matrix.shape

        sM = torch.nn.functional.softmax(sim_matrix.view(-1), dim=0)
        sM = sM.view(sim_matrix.shape)
        r = torch.sum(sM, dim=1)
        c = torch.sum(sM, dim=0)


        sim_matrix = -sim_matrix
        P = torch.exp(- lam * sim_matrix)
        P /= P.sum()
        u = torch.zeros(n, device=sim_matrix.device)
        cnt = 0
        # while np.max(np.abs(u - P.sum(1))) > epsilon:
        while cnt < niter:
            u = P.sum(1)
            P *= (r / u).reshape((-1, 1))
            P *= (c / P.sum(0)).reshape((1, -1))
            cnt += 1
        # return P, np.sum(P * sim_matrix), cnt
        return P
